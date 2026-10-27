import json
import pandas as pd
from pathlib import Path

import plotting_plotly as plotting

def config_to_string(cfg: dict) -> str:
    """
    Encode controller_config as:
      JC0, JS0, TS0, JS1, JS3, TS1, TS3, ...
    """

    dat = cfg.get("data_as_text", False)
    sep = cfg.get("data_separation", False)
    area = cfg.get("area_info", False)

    # First letter
    first = "T" if dat else "J"

    # Second letter
    second = "S" if sep else "C"

    # Radius / suffix
    if area:
        radius = cfg.get("area_radius")
        if radius is None:
            raise ValueError("area_info is True but area_radius is missing")
        suffix = str(radius)
    else:
        suffix = "0"

    return f"{first}{second}{suffix}"


def parse_result_file(path: Path) -> pd.DataFrame:
    """
    Parse one JSON result file into a pandas DataFrame
    with one row per run.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    rows = []

    data_block = raw.get("data", {})

    for model_key, model_data in data_block.items():
        # model name
        model = model_key.replace("_", "-")

        grid = model_data.get("grid_type")
        controller_cfg = model_data.get("controller_config", {})
        config_str = config_to_string(controller_cfg)

        for run_key, run_data in model_data.items():
            # skip summary keys
            if not run_key.startswith(model_key + "_run"):
                continue

            tc = run_data.get("tool_counter", {})

            row = {
                "model": model,
                "grid": grid,
                "bias": run_data.get("start_bias"),
                "config": config_str,
                "run_id": run_key.split("run")[-1],
                "success": run_data.get("voltage_in_bounds"),
                "error": "ERROR" in run_data,
                "tool_count": (
                    tc.get("get_measurements", 0)
                    + tc.get("adjust_generator", 0)
                    + tc.get("adjust_load", 0)
                ),
                "slack_power": run_data.get("power_at_slack"),

                "tc_get_meas": tc.get("get_measurements", 0),
                "tc_adj_gen": tc.get("adjust_generator", 0),
                "tc_adj_gen_up": tc.get("adjust_generator_up", 0),
                "tc_adj_gen_down": tc.get("adjust_generator_down", 0),
                "tc_adj_gen_same": tc.get("adjust_generator_same", 0),
                "tc_adj_load": tc.get("adjust_load", 0),
                "tc_adj_load_up": tc.get("adjust_load_up", 0),
                "tc_adj_load_down": tc.get("adjust_load_down", 0),
                "tc_adj_load_same": tc.get("adjust_load_same", 0),
            }

            rows.append(row)

    return pd.DataFrame(rows)


def load_many(raw_data_dir: str | Path) -> pd.DataFrame:
    """
    Load all JSON files from a directory (non-recursive)
    and concatenate them into a single DataFrame.
    """
    raw_data_dir = Path(raw_data_dir)

    json_files = sorted(raw_data_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {raw_data_dir}")

    dfs = [parse_result_file(f) for f in json_files]
    return pd.concat(dfs, ignore_index=True)


def add_wrong_adj_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column `wrong_adj` to the DataFrame following the given logic.

    Rules:
    - If success == False → wrong_adj = NaN
    - bias in {-1, -2} → sum specific 'wrong' actions
    - bias in {1, 2} → sum opposite 'wrong' actions
    - bias == 0 → every adjustment is wrong
    """

    df = df.copy()

    # Helper to safely get columns (treat missing as 0)
    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else 0

    df["wrong_adj"] = pd.NA
    df["wrong_adj_pct"] = pd.NA

    # Totals for percent denominator (safe if columns missing)
    total_adj = (col("tc_adj_gen") + col("tc_adj_load")).astype("float")

    # bias == -1 or -2
    neg_bias = df["bias"].isin([-1, -2])
    wrong_neg = (
        col("tc_adj_load_up") +
        col("tc_adj_gen_down") +
        col("tc_adj_gen_same") +
        col("tc_adj_load_same")
    ).astype("float")
    df.loc[neg_bias, "wrong_adj"] = wrong_neg[neg_bias]

    # bias == 1 or 2
    pos_bias = df["bias"].isin([1, 2])
    wrong_pos = (
        col("tc_adj_load_down") +
        col("tc_adj_gen_up") +
        col("tc_adj_gen_same") +
        col("tc_adj_load_same")
    ).astype("float")
    df.loc[pos_bias, "wrong_adj"] = wrong_pos[pos_bias]

    # bias == 0 (every adjustment is wrong)
    zero_bias = (df["bias"] == 0)
    wrong_zero = (col("tc_adj_gen") + col("tc_adj_load")).astype("float")
    df.loc[zero_bias, "wrong_adj"] = wrong_zero[zero_bias]

    # bias == 3 (every adjustment might be correct)
    df.loc[(df["bias"] == 3), "wrong_adj"] = 0

    # Percent wrong (only for successful runs with a positive denominator)
    wrong_adj_num = pd.to_numeric(df["wrong_adj"], errors="coerce")
    denom_ok = total_adj.notna()

    df.loc[denom_ok, "wrong_adj_pct"] = 100.0 * (wrong_adj_num[denom_ok] / total_adj[denom_ok])
    df["wrong_adj_pct"] = df["wrong_adj_pct"].fillna(0)

    return df


if "__main__" == __name__:

    raw_data = load_many("analyse-grids/print_data/raw_data")

    raw_data_without_errors = raw_data.loc[raw_data["error"] == False]
    raw_data_without_errors = add_wrong_adj_column(raw_data_without_errors)

    raw_data_without_errors.to_csv("raw_data.csv")
    agg = (
        raw_data_without_errors.groupby(["config", "run_id"])
            .agg(
                avg_success=("success", "mean"),
                avg_tool_count=("tool_count", "mean"),
                avg_slack_power=("slack_power", "mean"),
                avg_wrong_adj=("wrong_adj", "mean"),
                avg_wrong_adj_pct=("wrong_adj_pct", "mean"),
            )
            .reset_index()
    )

    fig_hm = plotting.plot_sucess_heatmap(
        raw_data_without_errors,
        out="success_heatmap.pdf",
        template="publish5+plotly_white",
        title="",
        height=650
    )

    fig_bias = plotting.plot_bias_robustness(
        raw_data_without_errors,
        title="",
        out="bias_robustness_success.pdf",
        template="publish5+plotly_white",
        marker_size = 14,
        line_width=5,
        height=650
    )

    plot_wrong_adjustment_pct_by_bias = plotting.plot_avg_wrong_adjustment_pct_by_config(
        raw_data_without_errors,
        title="",
        out="wrong_adjustment_pct_by_bias.pdf",
        template="publish5+plotly_white",
        height=600
    )

    # optimal slack active power by bias -2 -1 ... 3
    optimal_semi = [1.0829828105, 0.57495027452, 0.24170747074, -0.38229263192, float("NaN"), -0.67216633251]
    optimal_rural = [0.86660902529, 0.27202115247, -0.079419776815, -1.4456073178, -1.5397089858, -1.592631198]

    raw_only_120 = raw_data_without_errors.loc[raw_data_without_errors["model"] == "gpt-oss-120"]

    fig_bias_slack = plotting.plot_bias_robustness_model_agg(
        raw_only_120,
        grid_to_optimal_slack={"rural1": optimal_rural, "semiurb4": optimal_semi},
        metric="slack_power",
        title="gpt-oss-120",
        out="bias_robustness_slack_gpt120.pdf",
        template="publish5+plotly_white",
        line_width=5,
        height=650,
        y_tickformat=None,
        metric_y="P in MW",
        marker_size = 14,
    )

    raw_data_without_errors_and_unsuccessfull = raw_data_without_errors.loc[raw_data_without_errors["success"] == True]

    agg_slack = (
        raw_data_without_errors_and_unsuccessfull.groupby(["config", "bias", "grid", "run_id"])
            .agg(
                avg_success=("success", "mean"),
                avg_tool_count=("tool_count", "mean"),
                avg_slack_power=("slack_power", "mean"),
                avg_wrong_adj=("wrong_adj", "mean"),
                avg_wrong_adj_pct=("wrong_adj_pct", "mean"),
            )
            .reset_index()
    )

    fig = plotting.violin_with_raw(agg[agg['avg_wrong_adj'].notna()], "avg_wrong_adj", y_title="Wrong adjustments", title="Wrong Adjustments", disable_zero_line=True, clip_on_percentile=98)
    fig.write_image("avg_wrong_adj.pdf")

    fig = plotting.violin_with_raw_double(agg, "avg_success", "avg_tool_count", y_title_top="success", y_title_bottom="Iteration count", disable_zero_line=True, template="publish5+plotly_white")
    fig.write_image("success_tool_violin.pdf")

    success_to_wrong_adj_pct = raw_data_without_errors.groupby(["success", "grid", "bias"]).agg(avg_wrong_adj_pct=("wrong_adj_pct", "mean"),avg_tool_count=("tool_count", "mean")).reset_index()

    model_grid_agg = raw_data_without_errors_and_unsuccessfull.groupby(["model", "grid"]).agg(avg_success=("success", "mean"),
                avg_tool_count=("tool_count", "mean"),
                avg_slack_power=("slack_power", "mean"),
                avg_wrong_adj=("wrong_adj", "mean"),
                avg_wrong_adj_pct=("wrong_adj_pct", "mean")).reset_index()
    model_grid_agg["experiment"] = model_grid_agg["model"] + "-" + model_grid_agg["grid"]


    category_labels = {
        "avg_success": "Success",
        "avg_tool_count": "Iterations<br> (inv.)",
        "avg_wrong_adj": "Wrong adj. (inv.)",
        "avg_wrong_adj_pct": "Wrong adj.<br> % (inv.)",
        "avg_slack_power": "Slack (inv.)",
    }
    plotting.radar_from_df(model_grid_agg, "experiment", metrics=["avg_success", "avg_tool_count", 
                                                                            "avg_slack_power",  "avg_wrong_adj", "avg_wrong_adj_pct"],
                                                                            higher_is_better=["success"],
                                                                            category_labels=category_labels)

    fig = plotting.plot_configs_by_model_and_grid_bars(raw_data_without_errors, metric_col="wrong_adj_pct", template="plotly_white+publish5", y_label="wrong adj. in %")
    fig.write_image("wrong_adj_bar.pdf")
    fig = plotting.plot_configs_by_model_and_grid_bars(raw_data_without_errors, metric_col="tool_count", template="plotly_white+publish5", y_label="iteration count", y_label_gap=-0.10)
    fig.write_image("tool_count_bar.pdf")
