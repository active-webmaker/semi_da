import os
from datetime import datetime, timedelta
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def load_env() -> None:
    """환경 변수 로드 (.env 파일이 있으면 불러옴)."""
    load_dotenv()


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE 계산 (0으로 나누는 경우를 방지)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    smape_val = 200 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])
    return smape_val


def load_data(base_dir: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """train / test / sample_submission 로드."""
    train_path = os.path.join(base_dir, "train", "train.csv")
    test_dir = os.path.join(base_dir, "test")
    sub_path = os.path.join(base_dir, "sample_submission.csv")

    train = pd.read_csv(train_path)
    train["영업일자"] = pd.to_datetime(train["영업일자"])

    test_files = {}
    if os.path.isdir(test_dir):
        for fname in sorted(os.listdir(test_dir)):
            if fname.lower().endswith(".csv"):
                fpath = os.path.join(test_dir, fname)
                df = pd.read_csv(fpath)
                df["영업일자"] = pd.to_datetime(df["영업일자"])
                test_files[fname] = df

    sample_submission = pd.read_csv(sub_path)

    return train, test_files, sample_submission


def basic_exploration(train: pd.DataFrame) -> None:
    """기술 통계량 및 포맷 확인 (터미널에 출력)."""
    print("[INFO] Train head:\n", train.head())
    print("[INFO] Train info:")
    print(train.info())
    print("[INFO] Train describe:\n", train.describe(include="all"))
    print("[INFO] 결측치 개수:\n", train.isnull().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 / 타입 이상 처리."""
    df = df.copy()

    # 매출수량: 결측이면 0으로 대체
    if "매출수량" in df.columns:
        df["매출수량"] = pd.to_numeric(df["매출수량"], errors="coerce").fillna(0).astype(int)

    # 범주형: 결측이면 'Unknown'으로 대체
    for col in ["영업장명_메뉴명"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """영업일자로부터 파생 변수 생성: 요일, 주차, 월, 계절, 휴일 여부(placeholder)."""
    df = df.copy()

    df["요일"] = df["영업일자"].dt.weekday
    df["주차"] = df["영업일자"].dt.isocalendar().week.astype(int)
    df["월"] = df["영업일자"].dt.month

    # 계절: 단순 분기 기준
    def season(month: int) -> int:
        if month in (12, 1, 2):
            return 0  # 겨울
        if month in (3, 4, 5):
            return 1  # 봄
        if month in (6, 7, 8):
            return 2  # 여름
        return 3      # 가을

    df["계절"] = df["월"].apply(season)

    # 휴일 여부: 상세 휴일 캘린더가 없으므로 주말 여부로 대체 (토/일)
    df["is_weekend"] = df["요일"].isin([5, 6]).astype(int)

    return df


def add_menu_features(df: pd.DataFrame) -> pd.DataFrame:
    """'영업장명_메뉴명'을 분리하고 단체/후식 여부 파생."""
    df = df.copy()

    if "영업장명_메뉴명" not in df.columns:
        return df

    place_col = []
    menu_col = []
    for v in df["영업장명_메뉴명"].astype(str):
        if "_" in v:
            p, m = v.split("_", 1)
        else:
            p, m = v, ""
        place_col.append(p)
        menu_col.append(m)

    df["영업장명"] = place_col
    df["메뉴명"] = menu_col

    # 단체 / 후식 여부 플래그
    df["is_group"] = df["영업장명_메뉴명"].str.contains("단체", na=False).astype(int)
    df["is_dessert"] = df["영업장명_메뉴명"].str.contains("후식", na=False).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """전체 특성 공학 파이프라인."""
    df = clean_data(df)
    df = add_date_features(df)
    df = add_menu_features(df)
    return df


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, cols) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """범주형 변수 레이블 인코딩. 학습된 인코더를 반환하여 테스트에도 동일 적용."""
    train = train.copy()
    test = test.copy()

    encoders: Dict[str, LabelEncoder] = {}
    for col in cols:
        le = LabelEncoder()
        # train + test를 합쳐 fit 후, train/test 각각 transform
        merged = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0).fillna("Unknown")
        le.fit(merged)

        train[col] = le.transform(train[col].astype(str).fillna("Unknown"))
        test[col] = le.transform(test[col].astype(str).fillna("Unknown"))
        encoders[col] = le

    return train, test, encoders


def prepare_train_valid(train: pd.DataFrame, valid_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """시간 순서를 유지한 train/valid 분리."""
    train_sorted = train.sort_values("영업일자").reset_index(drop=True)
    n = len(train_sorted)
    split_idx = int(n * (1 - valid_ratio))
    train_df = train_sorted.iloc[:split_idx].reset_index(drop=True)
    valid_df = train_sorted.iloc[split_idx:].reset_index(drop=True)
    return train_df, valid_df


def train_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols, target_col: str = "매출수량") -> RandomForestRegressor:
    """RandomForest 기반 회귀 모델 학습 및 검증 SMAPE 출력."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    x_train = train_df[feature_cols]
    y_train = train_df[target_col]
    x_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    model.fit(x_train, y_train)

    valid_pred = model.predict(x_valid)
    score = smape(y_valid.values, valid_pred)
    print(f"[INFO] Validation SMAPE: {score:.4f}")

    return model


def build_series_for_forecast(train: pd.DataFrame, test_files: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """train과 모든 test를 합쳐 최신 시계열 확보."""
    dfs = [train]
    for df in test_files.values():
        dfs.append(df[["영업일자", "영업장명_메뉴명", "매출수량"]])
    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    all_df["영업일자"] = pd.to_datetime(all_df["영업일자"])
    all_df = all_df.sort_values(["영업장명_메뉴명", "영업일자"]).reset_index(drop=True)
    return all_df


def naive_menu_level_forecast(all_df: pd.DataFrame, horizon: int = 7) -> Dict[str, np.ndarray]:
    """메뉴별 마지막 7일 평균을 이용한 단순 예측 (길이 horizon)."""
    forecasts: Dict[str, np.ndarray] = {}
    for menu, g in all_df.groupby("영업장명_메뉴명"):
        g_sorted = g.sort_values("영업일자")
        last_vals = g_sorted["매출수량"].tail(7).values
        if len(last_vals) == 0:
            mean_val = 0.0
        else:
            mean_val = float(np.mean(last_vals))
        forecasts[menu] = np.array([mean_val] * horizon)
    return forecasts


def fill_submission_with_forecast(sample_submission: pd.DataFrame, forecasts: Dict[str, np.ndarray], horizon: int = 7) -> pd.DataFrame:
    """sample_submission 포맷에 맞춰 예측값 채우기."""
    sub = sample_submission.copy()
    menu_cols = [c for c in sub.columns if c != "영업일자"]

    # 각 메뉴에 대해 horizon 길이의 예측 벡터가 있다고 가정하고, 행 순서대로 채움
    # TEST_xx+1~7일 구조에 대해 복잡한 매핑 정보가 없으므로, 동일한 7일 예측 패턴을 반복 적용.
    n_rows = len(sub)
    repeats = int(np.ceil(n_rows / horizon))

    for col in menu_cols:
        menu_name = col
        if menu_name in forecasts:
            pattern = forecasts[menu_name]
        else:
            # train/test에 없던 메뉴는 0으로 채움
            pattern = np.zeros(horizon, dtype=float)

        full_pred = np.tile(pattern, repeats)[:n_rows]
        sub[col] = full_pred

    return sub


def main() -> None:
    base_dir = os.getenv("DATA_DIR", os.getcwd())

    print("[INFO] 환경 변수 로드 중...")
    load_env()

    print("[INFO] 데이터 로드 중...")
    train, test_files, sample_submission = load_data(base_dir)

    print("[INFO] 기본 탐색 수행...")
    basic_exploration(train)

    print("[INFO] 특성 공학 수행 (train)...")
    train_fe = build_features(train)

    # train 전처리와 동일한 스키마를 갖는 test_all 구성 (인코딩 적용 위해)
    print("[INFO] 특성 공학 수행 (test 전체)...")
    test_list = []
    for name, df in test_files.items():
        df_copy = df.copy()
        df_copy["test_file"] = name
        test_list.append(df_copy)
    if len(test_list) > 0:
        test_all_raw = pd.concat(test_list, axis=0, ignore_index=True)
        test_fe = build_features(test_all_raw)
    else:
        test_fe = train_fe.iloc[0:0].copy()

    # 범주형 인코딩
    cat_cols = ["영업장명_메뉴명", "영업장명", "메뉴명"]
    cat_cols = [c for c in cat_cols if c in train_fe.columns]

    print("[INFO] 범주형 인코딩 수행...")
    train_enc, test_enc, encoders = encode_categoricals(train_fe, test_fe, cat_cols)

    # 모델 학습용 피처 선택
    feature_cols = [
        "요일", "주차", "월", "계절", "is_weekend",
        "영업장명_메뉴명", "영업장명", "메뉴명",
        "is_group", "is_dessert",
    ]
    feature_cols = [c for c in feature_cols if c in train_enc.columns]

    train_df, valid_df = prepare_train_valid(train_enc)
    print("[INFO] 모델 학습 중...")
    model = train_model(train_df, valid_df, feature_cols)

    # 추후 필요 시 모델 기반 예측으로 확장할 수 있으나,
    # 제출 포맷 복잡성을 고려해 현재는 메뉴별 단순 시계열 기반 예측을 사용.
    print("[INFO] 제출용 단순 시계열 예측 생성...")
    all_series = build_series_for_forecast(train, test_files)
    forecasts = naive_menu_level_forecast(all_series, horizon=7)
    submission = fill_submission_with_forecast(sample_submission, forecasts, horizon=7)

    output_path = os.path.join(base_dir, "submission.csv")
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] submission.csv 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
