import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_stage_log.txt")


def log(*args, **kwargs) -> None:
    """간단한 로깅 함수: 콘솔에 출력하고 동일한 내용을 로그 파일에도 저장."""
    text = " ".join(str(a) for a in args)
    # 콘솔 출력
    print(*args, **kwargs)
    # 파일에도 기록 (에러가 나더라도 학습이 멈추지 않도록 방어)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return 200 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])


def smape_competition_like(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """대회 평가식과 유사하게, 실제값이 0인 항목은 제외하고 SMAPE를 계산.

    공식 식은 업장별 가중치 w_s 등이 있지만 비공개이므로,
    여기서는 모든 (t, i) 중에서 A_{t,i} != 0인 샘플만 사용해 단순 평균을 계산한다.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denominator = (np.abs(y_true) + np.abs(y_pred))
    # 실제값이 0인 항목 제외 + 분모가 0인 항목 제거
    mask = (y_true != 0) & (denominator != 0)
    if not np.any(mask):
        return 0.0
    return 200 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "매출수량" in df.columns:
        # 숫자로 변환 후, 음수 및 극단값을 정리
        df["매출수량"] = pd.to_numeric(df["매출수량"], errors="coerce").fillna(0)

        positives = df["매출수량"][df["매출수량"] > 0]
        if len(positives) > 0:
            p99 = positives.quantile(0.99)
            df["매출수량"] = df["매출수량"].clip(lower=0, upper=p99)
        else:
            df["매출수량"] = df["매출수량"].clip(lower=0)

        df["매출수량"] = df["매출수량"].astype(int)
    for col in ["영업장명_메뉴명"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df


def _season(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def build_feature_row(
    menu_full: str,
    date: pd.Timestamp,
    sales_history: List[float],
    menu_mean_train: Dict[str, float],
    global_mean_train: float,
    cat_encoders: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    if "_" in str(menu_full):
        place, menu = str(menu_full).split("_", 1)
    else:
        place, menu = str(menu_full), ""

    weekday = int(date.weekday())
    month = int(date.month)

    month_period = pd.cut(pd.Series([date.day]), bins=[0, 10, 20, 31], labels=[1, 2, 3]).iloc[0]
    # month_period is categorical label -> cast to int
    month_period_int = int(month_period) if month_period is not pd.NA else 0

    rolling_7 = float(np.mean(sales_history[-7:])) if len(sales_history) > 0 else 0.0
    rolling_30 = float(np.mean(sales_history[-30:])) if len(sales_history) > 0 else 0.0

    premium_keywords = ["스페셜", "프리미엄", "특선", "세트", "단체", "정식", "2인", "4인"]
    is_premium = int(any(k in menu for k in premium_keywords))

    def enc(col: str, val: str) -> int:
        if col not in cat_encoders:
            return 0
        mapping = cat_encoders[col]
        return int(mapping.get(val, mapping.get("Unknown", 0)))

    lag_1 = float(sales_history[-1]) if len(sales_history) > 0 else 0.0
    lag_7 = float(sales_history[-7]) if len(sales_history) >= 7 else 0.0
    if lag_7 > 0:
        lag_7_ratio = float(lag_1 / (lag_7 + 1e-5))
    else:
        lag_7_ratio = 0.0

    row: Dict[str, float] = {
        "요일": weekday,
        "주차": int(date.isocalendar().week),
        "월": month,
        "계절": _season(month),
        "is_weekend": int(weekday in (5, 6)),
        "month_period": month_period_int,
        "is_before_weekend": int(weekday == 4),
        "is_after_weekend": int(weekday == 0),
        "영업장명_메뉴명": enc("영업장명_메뉴명", str(menu_full)),
        "영업장명": enc("영업장명", place),
        "메뉴명": enc("메뉴명", menu),
        "is_group": int("단체" in str(menu_full)),
        "is_dessert": int("후식" in str(menu_full)),
        "menu_len": int(len(menu)),
        "menu_token_cnt": int(len(menu.split())) if menu else 0,
        "is_premium": is_premium,
        "rolling_7_mean": rolling_7,
        "rolling_30_mean": rolling_30,
        "rolling_ratio": float(rolling_7 / (rolling_30 + 1e-5)),
        "메뉴_인코딩": float(menu_mean_train.get(str(menu_full), global_mean_train)),
        # lag 기반 피처
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_7_ratio": lag_7_ratio,
    }

    return row


def fit_two_stage_full(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[xgboost.XGBClassifier, xgboost.XGBRegressor]:
    X = train_df[feature_cols]
    y = train_df["매출수량"].values
    y_zero = (y == 0).astype(int)

    clf = xgboost.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    clf.fit(X, y_zero)

    reg = xgboost.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.75,
        colsample_bytree=1.0,
        gamma=0.0,
        random_state=42,
        n_jobs=-1,
    )

    nonzero_mask = y > 0
    if not np.any(nonzero_mask):
        raise ValueError("학습 데이터에서 매출수량>0 샘플이 없어 2-stage 회귀 모델을 학습할 수 없습니다.")
    # 회귀 타깃에 log1p 변환 적용
    y_reg_train = np.log1p(y[nonzero_mask])
    reg.fit(X.loc[nonzero_mask], y_reg_train)
    return clf, reg


def forecast_7days_for_test(
    test_df_raw: pd.DataFrame,
    menu_cols: List[str],
    clf: xgboost.XGBClassifier,
    reg: xgboost.XGBRegressor,
    feature_cols: List[str],
    menu_mean_train: Dict[str, float],
    global_mean_train: float,
    cat_encoders: Dict[str, Dict[str, int]],
    zero_threshold: float = 0.5,
    horizon: int = 7,
) -> pd.DataFrame:
    test_df = test_df_raw.copy()
    test_df["영업일자"] = pd.to_datetime(test_df["영업일자"], errors="coerce")
    test_df = test_df.sort_values(["영업장명_메뉴명", "영업일자"]).reset_index(drop=True)

    last_date = pd.to_datetime(test_df["영업일자"].max())
    if pd.isna(last_date):
        raise ValueError("test_df에서 영업일자 파싱에 실패했습니다.")

    # sales history per menu (actuals)
    hist_map: Dict[str, List[float]] = {}
    for menu, g in test_df.groupby("영업장명_메뉴명"):
        hist_map[str(menu)] = g["매출수량"].astype(float).tolist()

    preds = pd.DataFrame(index=range(1, horizon + 1), columns=menu_cols, dtype=float)

    for d in range(1, horizon + 1):
        cur_date = last_date + pd.Timedelta(days=d)

        rows = []
        menus_for_rows = []
        for menu_full in menu_cols:
            sales_hist = hist_map.get(str(menu_full), [])
            row = build_feature_row(
                menu_full=str(menu_full),
                date=cur_date,
                sales_history=sales_hist,
                menu_mean_train=menu_mean_train,
                global_mean_train=global_mean_train,
                cat_encoders=cat_encoders,
            )
            rows.append(row)
            menus_for_rows.append(str(menu_full))

        X = pd.DataFrame(rows)
        # ensure all features exist
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_cols]

        proba_zero = clf.predict_proba(X)[:, 1]
        # 회귀 예측(log 스케일)을 원래 스케일로 복원
        pred_reg_log = reg.predict(X)
        pred_reg = np.expm1(pred_reg_log)

        pred = pred_reg.copy()
        pred[proba_zero >= zero_threshold] = 0.0

        for menu_full, yhat in zip(menus_for_rows, pred):
            preds.loc[d, menu_full] = float(yhat)
            hist_map.setdefault(menu_full, []).append(float(yhat))

    return preds


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["요일"] = df["영업일자"].dt.weekday
    df["주차"] = df["영업일자"].dt.isocalendar().week.astype(int)
    df["월"] = df["영업일자"].dt.month

    def season(month: int) -> int:
        if month in (12, 1, 2):
            return 0
        if month in (3, 4, 5):
            return 1
        if month in (6, 7, 8):
            return 2
        return 3

    df["계절"] = df["월"].apply(season)
    df["is_weekend"] = df["요일"].isin([5, 6]).astype(int)

    month_period = pd.cut(df["영업일자"].dt.day, bins=[0, 10, 20, 31], labels=[1, 2, 3])
    df["month_period"] = month_period.cat.codes.add(1).astype(int)

    df["is_before_weekend"] = (df["요일"] == 4).astype(int)
    df["is_after_weekend"] = (df["요일"] == 0).astype(int)

    return df


def add_menu_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "영업장명_메뉴명" not in df.columns:
        return df

    place_col, menu_col = [], []
    for v in df["영업장명_메뉴명"].astype(str):
        if "_" in v:
            p, m = v.split("_", 1)
        else:
            p, m = v, ""
        place_col.append(p)
        menu_col.append(m)

    df["영업장명"] = place_col
    df["메뉴명"] = menu_col
    df["is_group"] = df["영업장명_메뉴명"].str.contains("단체", na=False).astype(int)
    df["is_dessert"] = df["영업장명_메뉴명"].str.contains("후식", na=False).astype(int)

    df["menu_len"] = df["메뉴명"].astype(str).str.len()
    df["menu_token_cnt"] = df["메뉴명"].astype(str).str.split().str.len()

    premium_keywords = ["스페셜", "프리미엄", "특선", "세트", "단체", "정식", "2인", "4인"]
    df["is_premium"] = df["메뉴명"].astype(str).str.contains("|".join(premium_keywords), na=False).astype(int)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["영업장명_메뉴명", "영업일자"]).copy()

    df["rolling_7_mean"] = df.groupby("영업장명_메뉴명")["매출수량"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df["rolling_30_mean"] = df.groupby("영업장명_메뉴명")["매출수량"].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    df["rolling_7_mean"] = df.groupby("영업장명_메뉴명")["rolling_7_mean"].shift(1)
    df["rolling_30_mean"] = df.groupby("영업장명_메뉴명")["rolling_30_mean"].shift(1)

    df["rolling_7_mean"] = df["rolling_7_mean"].fillna(0)
    df["rolling_30_mean"] = df["rolling_30_mean"].fillna(0)

    df["rolling_ratio"] = df["rolling_7_mean"] / (df["rolling_30_mean"] + 1e-5)

    # lag 기반 피처: 직전 하루 및 7일 전 매출
    df["lag_1"] = df.groupby("영업장명_메뉴명")["매출수량"].shift(1).fillna(0)
    df["lag_7"] = df.groupby("영업장명_메뉴명")["매출수량"].shift(7).fillna(0)
    df["lag_7_ratio"] = df["lag_1"] / (df["lag_7"] + 1e-5)

    return df


def add_avg_features(df: pd.DataFrame) -> pd.DataFrame:
    menu_mean = df.groupby("영업장명_메뉴명")["매출수량"].mean()

    df["메뉴_인코딩"] = df["영업장명_메뉴명"].map(menu_mean)

    global_mean = df["매출수량"].mean()
    df["메뉴_인코딩"] = df["영업장명_메뉴명"].map(menu_mean).fillna(global_mean)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = add_date_features(df)
    df = add_menu_features(df)
    df = add_rolling_features(df)
    df = add_avg_features(df)
    return df


def prepare_train_valid(train_df: pd.DataFrame, valid_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_sorted = train_df.sort_values("영업일자").reset_index(drop=True)
    n = len(train_sorted)
    split_idx = int(n * (1 - valid_ratio))
    return train_sorted.iloc[:split_idx].copy(), train_sorted.iloc[split_idx:].copy()


def encode_train_only(
    train_df: pd.DataFrame,
    test_dfs: List[pd.DataFrame],
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, List[pd.DataFrame], Dict[str, Dict[str, int]]]:
    train_enc = train_df.copy()
    test_enc_list = [d.copy() for d in test_dfs]
    encoders: Dict[str, Dict[str, int]] = {}

    for col in cat_cols:
        train_vals = train_df[col].astype(str).fillna("Unknown")

        classes = pd.Index(train_vals.unique())
        if "Unknown" not in classes:
            classes = classes.append(pd.Index(["Unknown"]))

        class_to_int = {c: i for i, c in enumerate(classes)}

        train_enc[col] = train_vals.map(class_to_int).astype(int)

        for i in range(len(test_enc_list)):
            test_vals = test_dfs[i][col].astype(str).fillna("Unknown")
            test_enc_list[i][col] = test_vals.map(lambda x: class_to_int.get(x, class_to_int["Unknown"]))

        encoders[col] = class_to_int

    return train_enc, test_enc_list, encoders


def train_two_stage(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    valid_ratio: float = 0.1,
    zero_threshold_candidates: Optional[np.ndarray] = None,
) -> Tuple[xgboost.XGBClassifier, xgboost.XGBRegressor, pd.DataFrame, pd.DataFrame, float]:
    train_split, valid_split = prepare_train_valid(train_df, valid_ratio=valid_ratio)

    X_train = train_split[feature_cols]
    y_train = train_split["매출수량"].values

    X_valid = valid_split[feature_cols]
    y_valid = valid_split["매출수량"].values

    y_train_zero = (y_train == 0).astype(int)

    clf = xgboost.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train_zero)

    reg = xgboost.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.75,
        colsample_bytree=1.0,
        gamma=0.0,
        random_state=42,
        n_jobs=-1,
    )

    nonzero_mask = y_train > 0
    if np.any(nonzero_mask):
        # 회귀 타깃에 log1p 변환 적용
        y_reg_train = np.log1p(y_train[nonzero_mask])
        reg.fit(X_train.loc[nonzero_mask], y_reg_train)
    else:
        raise ValueError("학습 데이터에서 매출수량>0 샘플이 없어 2-stage 회귀 모델을 학습할 수 없습니다.")

    # 1차: 0.1 간격으로 threshold 탐색
    if zero_threshold_candidates is None:
        zero_threshold_candidates = np.arange(0.1, 0.91, 0.1)

    best_thr = 0.5
    best_smape = float("inf")

    log("[VALID] Threshold sweep (coarse):")
    for thr in zero_threshold_candidates:
        pred_valid_tmp = predict_two_stage(clf, reg, X_valid, zero_threshold=float(thr))
        smape_score = smape(y_valid, pred_valid_tmp)
        log(f"  thr={thr:.2f} -> SMAPE={smape_score:.4f}")
        if smape_score < best_smape:
            best_smape = smape_score
            best_thr = float(thr)

    # 2차: 1차에서 고른 best_thr 주변을 0.01 간격으로 정밀 탐색
    fine_start = max(best_thr - 0.05, 0.0)
    fine_end = min(best_thr + 0.05, 1.0)
    fine_grid = np.arange(fine_start, fine_end + 1e-8, 0.01)

    log("[VALID] Threshold sweep (fine):")
    for thr in fine_grid:
        pred_valid_tmp = predict_two_stage(clf, reg, X_valid, zero_threshold=float(thr))
        smape_score = smape(y_valid, pred_valid_tmp)
        log(f"  thr={thr:.2f} -> SMAPE={smape_score:.4f}")
        if smape_score < best_smape:
            best_smape = smape_score
            best_thr = float(thr)

    # 최종 best_thr 기준으로 검증 지표 출력
    pred_valid = predict_two_stage(clf, reg, X_valid, zero_threshold=best_thr)

    log(f"[VALID] Best zero_threshold: {best_thr:.4f}")
    log("[VALID] SMAPE_all:", f"{smape(y_valid, pred_valid):.4f}")
    log("[VALID] SMAPE_competition:", f"{smape_competition_like(y_valid, pred_valid):.4f}")
    log("[VALID] RMSE:", f"{np.sqrt(mean_squared_error(y_valid, pred_valid)):.4f}")
    log("[VALID] MAE:", f"{mean_absolute_error(y_valid, pred_valid):.4f}")

    pred_valid_int = np.clip(np.rint(pred_valid), 0, None)
    log("[VALID] SMAPE_all (int):", f"{smape(y_valid, pred_valid_int):.4f}")
    log("[VALID] SMAPE_competition (int):", f"{smape_competition_like(y_valid, pred_valid_int):.4f}")
    log("[VALID] RMSE (int):", f"{np.sqrt(mean_squared_error(y_valid, pred_valid_int)):.4f}")
    log("[VALID] MAE (int):", f"{mean_absolute_error(y_valid, pred_valid_int):.4f}")

    return clf, reg, train_split, valid_split, best_thr


def predict_two_stage(
    clf: xgboost.XGBClassifier,
    reg: xgboost.XGBRegressor,
    X: pd.DataFrame,
    zero_threshold: float = 0.5,
) -> np.ndarray:
    proba_zero = clf.predict_proba(X)[:, 1]
    # 회귀 예측(log 스케일)을 원래 스케일로 복원
    pred_reg_log = reg.predict(X)
    pred_reg = np.expm1(pred_reg_log)

    pred = pred_reg.copy()
    pred[proba_zero >= zero_threshold] = 0.0
    return pred


def load_data(base_dir: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    base_dir = os.path.abspath(base_dir)
    train_path = os.path.join(base_dir, "train", "train.csv")
    test_dir = os.path.join(base_dir, "test")
    sub_path = os.path.join(base_dir, "sample_submission.csv")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            "train.csv not found. "
            f"Expected: {train_path}. "
            "Set DATA_DIR to the folder that contains train/, test/, sample_submission.csv."
        )
    if not os.path.isfile(sub_path):
        raise FileNotFoundError(
            "sample_submission.csv not found. "
            f"Expected: {sub_path}. "
            "Set DATA_DIR to the folder that contains train/, test/, sample_submission.csv."
        )

    train = pd.read_csv(train_path)
    train["영업일자"] = pd.to_datetime(train["영업일자"])

    test_files: Dict[str, pd.DataFrame] = {}
    if os.path.isdir(test_dir):
        for fname in sorted(os.listdir(test_dir)):
            if fname.lower().endswith(".csv"):
                fpath = os.path.join(test_dir, fname)
                df = pd.read_csv(fpath)
                df["영업일자"] = pd.to_datetime(df["영업일자"])
                test_files[fname] = df

    sample_submission = pd.read_csv(sub_path)
    return train, test_files, sample_submission


def main() -> None:
    load_dotenv()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.getenv("DATA_DIR", script_dir)

    train, test_files, sample_submission = load_data(base_dir)

    train_fe = build_features(train)

    test_list = []
    for name, df in test_files.items():
        df_copy = df.copy()
        df_copy["test_file"] = name
        test_list.append(df_copy)

    for i, item in enumerate(test_list):
        test_list[i] = build_features(item)

    cat_cols = [c for c in ["영업장명_메뉴명", "영업장명", "메뉴명"] if c in train_fe.columns]

    train_enc, test_enc_list, encoders = encode_train_only(train_fe, test_list, cat_cols)

    menu_mean_train = train.groupby("영업장명_메뉴명")["매출수량"].mean().to_dict()
    global_mean_train = float(train["매출수량"].mean())

    feature_cols = [
        "요일",
        "월",
        "계절",
        "is_weekend",
        "영업장명_메뉴명",
        "메뉴명",
        "is_group",
        "is_dessert",
        "rolling_7_mean",
        "rolling_30_mean",
        "메뉴_인코딩",
        "month_period",
        "is_before_weekend",
        "is_after_weekend",
        "menu_len",
        "menu_token_cnt",
        "is_premium",
        "rolling_ratio",
        "lag_1",
        "lag_7",
        "lag_7_ratio",
    ]
    feature_cols = [c for c in feature_cols if c in train_enc.columns]

    # (선택) 검증 출력 + zero_threshold 탐색
    _clf_tmp, _reg_tmp, _train_split, _valid_split, best_zero_threshold = train_two_stage(
        train_enc,
        feature_cols,
        valid_ratio=0.1,
    )

    # 제출용: 전체 train으로 재학습
    clf, reg = fit_two_stage_full(train_enc, feature_cols)

    for idx, df in enumerate(test_enc_list):
        X_test = df[feature_cols]
        y_true = df["매출수량"].values

        pred = predict_two_stage(clf, reg, X_test, zero_threshold=best_zero_threshold)

        log(f"[TEST {idx}] SMAPE_all:", f"{smape(y_true, pred):.4f}")
        log(f"[TEST {idx}] SMAPE_competition:", f"{smape_competition_like(y_true, pred):.4f}")
        log(f"[TEST {idx}] RMSE:", f"{np.sqrt(mean_squared_error(y_true, pred)):.4f}")
        log(f"[TEST {idx}] MAE:", f"{mean_absolute_error(y_true, pred):.4f}")

        pred_int = np.clip(np.rint(pred), 0, None)
        log(f"[TEST {idx}] SMAPE_all (int):", f"{smape(y_true, pred_int):.4f}")
        log(f"[TEST {idx}] SMAPE_competition (int):", f"{smape_competition_like(y_true, pred_int):.4f}")
        log(f"[TEST {idx}] RMSE (int):", f"{np.sqrt(mean_squared_error(y_true, pred_int)):.4f}")
        log(f"[TEST {idx}] MAE (int):", f"{mean_absolute_error(y_true, pred_int):.4f}")

    # submission.csv 생성 (TEST_00~09 각각의 마지막 날짜 이후 7일 예측)
    sub = sample_submission.copy()
    menu_cols = [c for c in sub.columns if c != "영업일자"]

    forecasts_by_test: Dict[str, pd.DataFrame] = {}
    for test_id in range(10):
        test_name = f"TEST_{test_id:02d}.csv"
        if test_name not in test_files:
            continue
        preds_7 = forecast_7days_for_test(
            test_df_raw=test_files[test_name],
            menu_cols=menu_cols,
            clf=clf,
            reg=reg,
            feature_cols=feature_cols,
            menu_mean_train=menu_mean_train,
            global_mean_train=global_mean_train,
            cat_encoders=encoders,
            zero_threshold=best_zero_threshold,
            horizon=7,
        )
        forecasts_by_test[f"TEST_{test_id:02d}"] = preds_7

    for i in range(len(sub)):
        key = str(sub.loc[i, "영업일자"])
        if "+" not in key:
            continue
        test_key, offset_str = key.split("+", 1)
        day = int(offset_str.replace("일", ""))
        if test_key not in forecasts_by_test:
            continue
        row_pred = forecasts_by_test[test_key].loc[day]
        sub.loc[i, menu_cols] = row_pred.values

    # 제출 포맷: 정수화 + 음수 방지
    sub[menu_cols] = sub[menu_cols].astype(float).round().clip(lower=0).astype(int)

    output_path = os.path.join(base_dir, "submission.csv")
    sub.to_csv(output_path, index=False, encoding="utf-8-sig")
    log(f"[INFO] submission saved: {output_path}")


if __name__ == "__main__":
    main()
