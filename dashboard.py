import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Bank Churn Dashboard", layout="wide")

# Shared plotting palette (from your screenshot)
PALETTE_LIGHT = "#9FD6FF"  # light blue
PALETTE_PINK = "#F7B7C6"   # light pink
PALETTE_ORANGE = "#E3893B"  # warm accent for 3-color charts


def chart_colors(n: int) -> list[str]:
	"""Return a list of n colors following the requested rules:
	- n == 1: use light blue
	- n == 2: use light blue, light pink
	- n >= 3: use light blue, light pink, warm accent (repeat if needed)
	"""
	base = [PALETTE_LIGHT, PALETTE_PINK, PALETTE_ORANGE]
	if n <= 0:
		return [PALETTE_LIGHT]
	if n == 1:
		return [PALETTE_LIGHT]
	if n == 2:
		return [PALETTE_LIGHT, PALETTE_PINK]
	# for 3+ keep base order, repeat when needed
	colors = []
	i = 0
	while len(colors) < n:
		colors.append(base[i % len(base)])
		i += 1
	return colors


def get_theme(mode: str) -> dict:
	if mode == "Dark":
		return {
			"bg": "radial-gradient(circle at top left, #101722 0%, #0f1621 45%, #0c111a 100%)",
			"surface": "#161f2d",
			"surface_alt": "#1d293a",
			"chart_bg": "#111a28",
			"border": "#33445b",
			"text": "#eaf1fb",
			"text_muted": "#b4c1d6",
			"accent_1": "#9FD6FF",
			"accent_2": "#f39b46",
			"grid": "#304158",
			"axis": "#4c607e",
			"toggle_bg": "#1f2a3a",
			"toggle_fg": "#f5f7ff",
		}

	return {
		"bg": "radial-gradient(circle at top left, #f6f3ef 0%, #f0ece7 48%, #ede8e3 100%)",
		"surface": "#f7f4f1",
		"surface_alt": "#fbfaf8",
		"chart_bg": "#ffffff",
		"border": "#d8d2ca",
		"text": "#1d232f",
		"text_muted": "#6b7280",
		"accent_1": "#9FD6FF",
		"accent_2": "#E3893B",
		"grid": "#d8dee7",
		"axis": "#9aa4b2",
		"toggle_bg": "#ffffff",
        "toggle_fg": "#FFD54F",
	}


@st.cache_data
def load_data() -> pd.DataFrame:
	data_path = "data/cleaned_data.csv"
	df = pd.read_csv(data_path)
	df.columns = [c.lower() for c in df.columns]

	# Keep customer id and surname for detailed tables when available.

	if "geography" in df.columns:
		df["geography"] = df["geography"].astype(str).str.lower().str.strip()
	if "gender" in df.columns:
		df["gender"] = df["gender"].astype(str).str.lower().str.strip()

	if "saving" not in df.columns:
		df["saving"] = (df["balance"] > 0).astype(int)

	if "estimatedsalary_group" not in df.columns:
		df["estimatedsalary_group"] = pd.cut(
			df["estimatedsalary"],
			bins=[0, 1000, 20000, 50000, 100000, 150000, 200000, 400000],
			labels=[
				"below 1k",
				"1k - 20k",
				"20k - 50k",
				"50k - 100k",
				"100k - 150k",
				"150k - 200k",
				"above 200k",
			],
			right=False,
		)

	if "age_group" not in df.columns:
		df["age_group"] = pd.cut(
			df["age"],
			bins=[17, 20, 23, 31, 44, 60, 120],
			labels=["18-20", "21-23", "24-31", "32-44", "45-60", "above 60"],
		)

	if "saving_group" not in df.columns:
		df["saving_group"] = pd.cut(
			df["balance"],
			bins=[-1, 0, 50000, 100000, 150000, 200000, 300000],
			labels=[
				"no saving",
				"0-50k",
				"50k-100k",
				"100k - 150k",
				"150k - 200k",
				"above 200k",
			],
		)

	if "creditscore_group" not in df.columns:
		df["creditscore_group"] = pd.cut(
			df["creditscore"],
			bins=[-np.inf, 650, 700, 800, np.inf],
			labels=["below 650", "650-700", "700-800", "above 800"],
			include_lowest=True,
		)

	df["credit_group_3"] = pd.cut(
		df["creditscore"],
		bins=[-np.inf, 650, 700, np.inf],
		labels=["below 650", "650-700", "above 700"],
		include_lowest=True,
	)

	return df


def _slugify(value: str) -> str:
	return "".join(character if character.isalnum() else "_" for character in str(value).lower()).strip("_")


def _filter_key(name: str, suffix: str, prefix: str = "") -> str:
	prefix_value = f"{_slugify(prefix)}_" if prefix else ""
	return f"filter_{prefix_value}{_slugify(name)}_{suffix}"


def _label_with_k(label: str) -> str:
	text = str(label)
	normalized = text.lower().replace(" ", "")
	mapping = {
		"50k-1lakh": "50k-100k",
		"1lakh-1.5lakh": "100k-150k",
		"1.5lakh-2lakh": "150k-200k",
		"above2lakh": "above 200k",
		"1lakh-2lakh": "100k-200k",
	}
	return mapping.get(normalized, text)


def _sync_checkbox_filter(name: str, options: list[str], prefix: str = "") -> None:
	selected = [
		option
		for option in options
		if st.session_state.get(_filter_key(name, _slugify(option), prefix), False)
	]
	st.session_state[_filter_key(name, "selected", prefix)] = selected
	st.session_state[_filter_key(name, "all", prefix)] = len(selected) == len(options)


def _sync_checkbox_filter_all(name: str, options: list[str], prefix: str = "") -> None:
	all_checked = st.session_state.get(_filter_key(name, "all", prefix), False)
	for option in options:
		st.session_state[_filter_key(name, _slugify(option), prefix)] = all_checked
	st.session_state[_filter_key(name, "selected", prefix)] = list(options) if all_checked else []


def checkbox_filter_group(label: str, options: list[str], default_all: bool = True, key_prefix: str = "") -> list[str]:
	selected_key = _filter_key(label, "selected", key_prefix)
	all_key = _filter_key(label, "all", key_prefix)

	if selected_key not in st.session_state:
		st.session_state[selected_key] = list(options) if default_all else []
	if all_key not in st.session_state:
		st.session_state[all_key] = default_all
	for option in options:
		option_key = _filter_key(label, _slugify(option), key_prefix)
		if option_key not in st.session_state:
			st.session_state[option_key] = default_all

	selected = st.session_state[selected_key]
	count = len(selected)

	with st.expander(f"{label} ({count} selected)", expanded=False):
		st.markdown(f'<div class="filter-name">{label}</div>', unsafe_allow_html=True)
		st.markdown(f'<div class="filter-count">({count} selected)</div>', unsafe_allow_html=True)
		st.checkbox("All", key=all_key, on_change=_sync_checkbox_filter_all, args=(label, options, key_prefix))
		for option in options:
			st.checkbox(
				option,
				key=_filter_key(label, _slugify(option), key_prefix),
				on_change=_sync_checkbox_filter,
				args=(label, options, key_prefix),
			)

	return selected


def _set_detail_chart_view(option: str) -> None:
	st.session_state.detail_chart_view = option


def render_detail_chart_switcher() -> str:
	options = [
		("Credit Score", "Credit Score Group"),
		("Saving", "Saving Group"),
		("Income", "Income Group"),
	]
	if "detail_chart_view" not in st.session_state:
		st.session_state.detail_chart_view = options[0][1]

	button_cols = st.columns(3)
	for index, (label, value) in enumerate(options):
		with button_cols[index]:
			button_type = "primary" if st.session_state.detail_chart_view == value else "secondary"
			st.button(
				label,
				key=f"detail_chart_{_slugify(value)}",
				use_container_width=True,
				type=button_type,
				on_click=_set_detail_chart_view,
				args=(value,),
			)

	return st.session_state.detail_chart_view


def apply_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
	st.markdown("### Filters", unsafe_allow_html=True)
	geo_sel = checkbox_filter_group(
		"Geography",
		sorted(df["geography"].dropna().astype(str).unique().tolist()),
		key_prefix=key_prefix,
	)
	gender_sel = checkbox_filter_group(
		"Gender",
		sorted(df["gender"].dropna().astype(str).unique().tolist()),
		key_prefix=key_prefix,
	)
	active_sel = checkbox_filter_group("Active Status", ["Active", "Inactive"], key_prefix=key_prefix)
	churn_sel = checkbox_filter_group("Churn Status", ["Retained", "Churned"], key_prefix=key_prefix)
	product_sel = checkbox_filter_group("Products Holding", ["1", "2", "3", "4"], key_prefix=key_prefix)
	age_sel = checkbox_filter_group(
		"Age Group",
		["18-20", "21-23", "24-31", "32-44", "45-60", "above 60"],
		key_prefix=key_prefix,
	)
	credit_sel = checkbox_filter_group(
		"Credit Score Group",
		["below 650", "650-700", "700-800", "above 800"],
		key_prefix=key_prefix,
	)
	saving_sel = checkbox_filter_group(
		"Saving Group",
		["no saving", "0-50k", "50k-1 lakh", "1 lakh - 1.5 lakh", "1.5 lakh - 2 lakh", "above 2 lakh"],
		key_prefix=key_prefix,
	)
	income_sel = checkbox_filter_group(
		"Income Group",
		["below 1k", "1k - 20k", "20k - 50k", "50k - 1 lakh", "1 lakh - 1.5 lakh", "1.5 lakh - 2 lakh", "above 2 lakh"],
		key_prefix=key_prefix,
	)

	out = df.copy()
	out = out[out["geography"].isin(geo_sel)]
	out = out[out["gender"].isin(gender_sel)]
	out = out[out["age_group"].astype(str).isin(age_sel)]
	out = out[out["creditscore_group"].astype(str).isin(credit_sel)]
	out = out[out["saving_group"].astype(str).isin(saving_sel)]
	out = out[out["estimatedsalary_group"].astype(str).isin(income_sel)]

	if len(active_sel) != 2:
		if "Active" in active_sel and "Inactive" not in active_sel:
			out = out[out["isactivemember"] == 1]
		elif "Inactive" in active_sel and "Active" not in active_sel:
			out = out[out["isactivemember"] == 0]
		elif not active_sel:
			out = out.iloc[0:0]

	if len(churn_sel) != 2:
		if "Retained" in churn_sel and "Churned" not in churn_sel:
			out = out[out["exited"] == 0]
		elif "Churned" in churn_sel and "Retained" not in churn_sel:
			out = out[out["exited"] == 1]
		elif not churn_sel:
			out = out.iloc[0:0]

	if len(product_sel) != 4:
		product_values = [int(value) for value in product_sel]
		out = out[out["numofproducts"].isin(product_values)]

	return out


def apply_potential_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
	st.markdown("### Filters", unsafe_allow_html=True)
	geo_sel = checkbox_filter_group(
		"Geography",
		sorted(df["geography"].dropna().astype(str).unique().tolist()),
		key_prefix=key_prefix,
	)
	gender_sel = checkbox_filter_group(
		"Gender",
		sorted(df["gender"].dropna().astype(str).unique().tolist()),
		key_prefix=key_prefix,
	)
	age_sel = checkbox_filter_group(
		"Age Group",
		["18-20", "21-23", "24-31", "32-44", "45-60", "above 60"],
		key_prefix=key_prefix,
	)

	filtered = df.copy()
	filtered = filtered[filtered["geography"].isin(geo_sel)]
	filtered = filtered[filtered["gender"].isin(gender_sel)]
	filtered = filtered[filtered["age_group"].astype(str).isin(age_sel)]
	return filtered


def style_page(theme: dict) -> None:
	css = """
	<style>
	:root {
		--bg: __BG__;
		--surface: __SURFACE__;
		--surface-alt: __SURFACE_ALT__;
		--chart-bg: __CHART_BG__;
		--border: __BORDER__;
		--text: __TEXT__;
		--text-muted: __TEXT_MUTED__;
		--accent-1: __ACCENT1__;
		--accent-2: __ACCENT2__;
		--toggle-bg: __TOGGLE_BG__;
		--toggle-fg: __TOGGLE_FG__;
	}
	.stApp {
		background: var(--bg);
		color: var(--text);
	}
	.main .block-container {
		padding-top: 0.9rem;
	}
	h1, h2, h3, h4, h5, h6, p, div, span, label {
		color: var(--text);
	}
	.title-band {
		border: 1px solid var(--border);
		border-radius: 18px;
		background: var(--surface);
		padding: 14px 16px;
		margin-bottom: 10px;
	}
	.kpi-wrap {
		border: 1px solid var(--border);
		border-radius: 16px;
		background: var(--surface-alt);
		padding: 16px 18px;
		text-align: left;
		min-height: 84px;
	}
	.kpi-label {
		font-size: 13px;
		font-weight: 600;
		color: var(--text-muted);
		line-height: 1.1;
	}
	.kpi-value {
		font-size: 26px;
		font-weight: 700;
		color: var(--text);
		line-height: 1.2;
	}
	.panel {
		border: 1px solid var(--border);
		border-radius: 18px;
		background: var(--surface);
		padding: 10px;
	}
	.findings {
		border: 2px solid var(--border);
		border-radius: 18px;
		background: var(--surface-alt);
		min-height: 620px;
		padding: 18px;
		font-size: 16px;
		color: var(--text);
		line-height: 1.5;
	}
	.findings p {
		margin-bottom: 14px;
	}
	.findings h3 {
		font-size: 1.5rem;
		margin: 0 0 16px 0;
		line-height: 1.15;
	}
	.chart-gap {
		height: 20px;
	}
	.pill-row {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
		justify-content: flex-start;
		margin-top: 10px;
		margin-bottom: 8px;
	}
	.pill {
		border: 2px solid var(--border);
		border-radius: 999px;
		padding: 8px 18px;
		background: var(--surface-alt);
		font-size: 14px;
	}
	.stButton > button[kind="primary"] {
		background: var(--accent-1);
		border-color: var(--accent-1);
		color: #ffffff;
	}
	.stButton > button[kind="secondary"] {
		background: var(--toggle-bg) !important;
		border-color: var(--border);
		color: var(--toggle-fg) !important;
	}
	/* Consistent button sizing */
	.stButton > button {
		min-height: 64px;
		padding: 12px 16px;
		font-size: 14px;
		font-weight: 600;
		line-height: 1.2;
		white-space: normal;
	}
	.filter-name {
		font-size: 14px;
		font-weight: 700;
		margin: 0 0 2px 0;
		color: var(--text);
	}
	.filter-count {
		font-size: 12px;
		font-weight: 600;
		margin: 0 0 8px 0;
		color: var(--text-muted);
	}
	.sub-pill {
		display: inline-block;
		border: 1px solid var(--border);
		border-radius: 999px;
		padding: 4px 10px;
		margin-right: 6px;
		margin-bottom: 6px;
		font-size: 12px;
		background: var(--surface-alt);
	}
	/* Form controls */
	[data-baseweb="select"] > div,
	[data-baseweb="input"] > div,
	.stMultiSelect [data-baseweb="tag"] {
		background: var(--surface-alt);
		border-color: var(--border);
		color: var(--text);
	}
	[data-baseweb="input"] input {
		color: var(--text) !important;
		-webkit-text-fill-color: var(--text) !important;
	}
	.stSelectbox label, .stMultiSelect label, .stRadio label, .stNumberInput label, .stTextInput label {
		color: var(--text);
	}
	/* Select dropdown menu styling */
	div[data-baseweb="popover"],
	div[data-baseweb="popover"] * {
		background: var(--surface-alt) !important;
		color: var(--text) !important;
	}
	div[data-baseweb="popover"] li:hover {
		background: var(--surface) !important;
	}
	.st-emotion-cache-16txtl3, .st-emotion-cache-1y4p8pa {
		color: var(--text);
	}

	div[data-testid="stButton"] > button:hover {
		border-color: var(--accent-1) !important;
		color: var(--accent-1) !important;
	}
	div[data-testid="stButton"] > button:focus,
	div[data-testid="stButton"] > button:active {
		outline: none !important;
		box-shadow: none !important;
	}
	/* Plotly chart cards: consistent rounded border and readable text */
	[data-testid="stPlotlyChart"] {
		border: 2px solid var(--border);
		border-radius: 18px;
		background: var(--chart-bg);
		padding: 8px 8px 2px 8px;
		margin-bottom: 22px;
		overflow: hidden !important;
	}
	[data-testid="stPlotlyChart"] > div {
		background: transparent !important;
		overflow: hidden !important;
	}
	[data-testid="stPlotlyChart"] iframe {
		overflow: hidden !important;
	}
	/* Dataframe styling to respect light/dark theme */
	.custom-table-wrapper {
		overflow-y: auto;
		border: 1px solid var(--border);
		border-radius: 14px;
		background: var(--surface);
	}
	.custom-dataframe {
		width: 100%;
		border-collapse: collapse;
		text-align: left;
	}
	.custom-dataframe th {
		position: sticky;
		top: 0;
		background: var(--surface-alt);
		color: var(--text);
		padding: 10px;
		border-bottom: 1px solid var(--border);
		font-weight: 600;
		font-size: 14px;
		z-index: 1;
	}
	.custom-dataframe td {
		padding: 8px 10px;
		border-bottom: 1px solid var(--border);
		color: var(--text);
		font-size: 13px;
	}
	.custom-dataframe tr:last-child td {
		border-bottom: none;
	}
	.custom-dataframe tr:hover {
		background: var(--surface-alt);
	}

	/* Hypothesis inline table inside panels */
	.hypothesis-table {
		width: 100%;
		border-collapse: collapse;
		margin-top: 6px;
	}
	.hypothesis-table th,
	.hypothesis-table td {
		padding: 8px 10px;
		border-bottom: 1px solid var(--border);
		color: var(--text) !important;
	}
	.hypothesis-table thead th {
		background: var(--surface-alt) !important;
		color: var(--text) !important;
		font-weight: 700;
	}
	</style>
	"""

	css = (
		css.replace("__BG__", theme["bg"])
		.replace("__SURFACE__", theme["surface"])
		.replace("__SURFACE_ALT__", theme["surface_alt"])
		.replace("__CHART_BG__", theme["chart_bg"])
		.replace("__BORDER__", theme["border"])
		.replace("__TEXT__", theme["text"])
		.replace("__TEXT_MUTED__", theme["text_muted"])
		.replace("__ACCENT1__", theme["accent_1"])
		.replace("__ACCENT2__", theme["accent_2"])
		.replace("__TOGGLE_BG__", theme["toggle_bg"])
		.replace("__TOGGLE_FG__", theme["toggle_fg"])
	)

	st.markdown(
		css,
		unsafe_allow_html=True,
	)


def render_theme_dataframe(df: pd.DataFrame, height: int = 400) -> None:
	display_df = df.head(1000)
	html_table = display_df.to_html(index=False, classes="custom-dataframe")
	st.markdown(
		f"""
		<div class="custom-table-wrapper" style="max-height: {height}px;">
			{html_table}
		</div>
		""",
		unsafe_allow_html=True,
	)
	if len(df) > 1000:
		st.caption(f"Showing first 1000 rows (out of {len(df)} total).")


def small_metric(label: str, value: str) -> None:
	st.markdown(
		f"""
		<div class="kpi-wrap">
			<div class="kpi-label">{label}</div>
			<div class="kpi-value">{value}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def format_abbrev(value: float) -> str:
	abs_value = abs(value)
	if abs_value >= 1_000_000:
		formatted = f"{value / 1_000_000:.1f}M"
	elif abs_value >= 1_000:
		formatted = f"{value / 1_000:.1f}K"
	else:
		formatted = f"{value:,.0f}"
	return formatted.replace(".0K", "K").replace(".0M", "M")


def style_figure(fig, theme: dict, height: int = 260) -> None:
	fig.update_layout(
		height=height,
		margin=dict(l=10, r=10, t=44, b=12),
		paper_bgcolor=theme["chart_bg"],
		plot_bgcolor=theme["chart_bg"],
		font=dict(color=theme["text"]),
		legend=dict(font=dict(color=theme["text"])),
		title_font=dict(color=theme["text"], size=13),
	)
	fig.update_xaxes(
		gridcolor=theme["grid"],
		linecolor=theme["axis"],
		tickfont=dict(color=theme["text"]),
		title_font=dict(color=theme["text"]),
	)
	fig.update_yaxes(
		gridcolor=theme["grid"],
		linecolor=theme["axis"],
		tickfont=dict(color=theme["text"]),
		title_font=dict(color=theme["text"]),
		title_text="",
		showticklabels=False,
	)


def toggle_theme() -> None:
	st.session_state.theme_mode = "Dark" if st.session_state.theme_mode == "Light" else "Light"


def render_navbar(theme_mode: str) -> None:
	if "current_page" not in st.session_state:
		st.session_state.current_page = "overview"

	left, center, right = st.columns([2.2, 5.6, 0.9], vertical_alignment="center")
	with left:
		st.markdown("<div style='padding-top:6px; font-size:34px; font-weight:700;'>Bank Churn Analysis</div>", unsafe_allow_html=True)
	with center:
		btn_cols = st.columns(5)
		pages = [
			("Customer Overview", "overview"),
			("Churn Analysis", "churn_analysis"),
			("Churn Prediction", "churn_prediction"),
			("Strong Engagement Customers", "strong_engagement"),
			("Potential Active Customers", "potential_active"),
		]
		for index, (label, page_key) in enumerate(pages):
			with btn_cols[index]:
				btn_type = "primary" if st.session_state.current_page == page_key else "secondary"
				st.button(
					label,
					use_container_width=True,
					type=btn_type,
					on_click=lambda key=page_key: st.session_state.update({"current_page": key}),
				)
	with right:
		icon = "\u263d" if theme_mode == "Light" else "\u2600"
		st.button(icon, key="theme_toggle", on_click=toggle_theme, type="secondary")


def render_overview(df: pd.DataFrame, theme: dict) -> None:
	st.markdown("## Customer Overview", unsafe_allow_html=True)
	st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
	col_a, col_b, col_c, col_d, col_e = st.columns(5)

	filtered_for_kpi = df.copy()
	total_customers = len(filtered_for_kpi)
	active_rate = (filtered_for_kpi["isactivemember"].mean() * 100) if total_customers else 0
	avg_credit = filtered_for_kpi["creditscore"].mean() if total_customers else 0
	positive_saving = filtered_for_kpi[filtered_for_kpi["balance"] > 0]
	avg_saving = positive_saving["balance"].mean() if not positive_saving.empty else 0
	avg_income = filtered_for_kpi["estimatedsalary"].mean() if total_customers else 0

	with col_a:
		small_metric("Total Customers", f"{total_customers:,}")
	with col_b:
		small_metric("Active Rate", f"{active_rate:,.1f}%")
	with col_c:
		small_metric("Avg Credit Score", f"{avg_credit:,.0f}")
	with col_d:
		small_metric("Avg Saving", format_abbrev(avg_saving))
	with col_e:
		small_metric("Avg Customer Income", format_abbrev(avg_income))

	st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
	left, center, right = st.columns([1.2, 5.2, 1.55], gap="large")

	with left:
		filtered = apply_filters(df, key_prefix="overview")

	with center:
		c1, c2 = st.columns(2)

		tenure_order = sorted(filtered["tenure"].dropna().unique().tolist())
		tenure_counts = filtered["tenure"].value_counts().reindex(tenure_order).fillna(0).reset_index()
		tenure_counts.columns = ["tenure", "customers"]
		tenure_counts["tenure_label"] = tenure_counts["tenure"].astype(int).astype(str) + " year"
		tenure_counts.loc[tenure_counts["tenure"] == 10, "tenure_label"] = "10 years"

		fig_tenure = px.line(
			tenure_counts,
			x="tenure_label",
			y="customers",
			markers=True,
			title="Customers by Tenure (Life Time)",
			color_discrete_sequence=chart_colors(1),
		)
		style_figure(fig_tenure, theme)
		with c1:
			st.markdown('<div style="height:0px;"></div>', unsafe_allow_html=True)
			st.plotly_chart(fig_tenure, use_container_width=True, config={"displayModeBar": False})

		potential_active_base = filtered[
			(filtered["hascrcard"] == 0)
			& (filtered["exited"] == 0)
			& (filtered["creditscore"] >= 700)
			& (filtered["estimatedsalary"] >= 100000)
			& (filtered["balance"] >= 100000)
		]
		with c2:
			show_potential_active = st.toggle(
				"Show potential active customers",
				value=False,
				key="toggle_potential_active_ccard",
			)
			ccard_source = potential_active_base if show_potential_active else filtered
			ccard = ccard_source.groupby(["hascrcard", "isactivemember"]).size().reset_index(name="customers")
			ccard["credit_card"] = ccard["hascrcard"].map({1: "Credit Card Holder", 0: "No Credit Card Holder"})
			ccard["status"] = ccard["isactivemember"].map({1: "Active", 0: "Inactive"})
			fig_card = px.bar(
				ccard,
				x="credit_card",
				y="customers",
				color="status",
				barmode="stack",
				title="Customers by Credit Card and Active Status",
				color_discrete_sequence=chart_colors(2),
			)
			style_figure(fig_card, theme, height=220)
			fig_card.update_layout(legend_title_text="")
			st.plotly_chart(fig_card, use_container_width=True, config={"displayModeBar": False})

		st.markdown('<div class="chart-gap"></div>', unsafe_allow_html=True)

		c3, c4 = st.columns(2)
		age_ct = filtered["age_group"].value_counts().sort_index().reset_index()
		age_ct.columns = ["age_group", "customers"]
		fig_age = px.bar(
			age_ct,
			x="age_group",
			y="customers",
			title="Customers by Age Group",
			color_discrete_sequence=chart_colors(1),
		)
		style_figure(fig_age, theme)
		fig_age.update_layout(showlegend=False)
		with c3:
			st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})

		geo_saving = filtered.groupby(["geography", "saving"]).size().reset_index(name="customers")
		geo_saving["saving_status"] = geo_saving["saving"].map({1: "Saving", 0: "Non Saving"})
		fig_geo = px.bar(
			geo_saving,
			x="geography",
			y="customers",
			color="saving_status",
			barmode="group",
			title="Customers by Saving Status and Geography",
			color_discrete_sequence=chart_colors(2),
		)
		style_figure(fig_geo, theme)
		fig_geo.update_layout(legend_title_text="")
		with c4:
			st.plotly_chart(fig_geo, use_container_width=True, config={"displayModeBar": False})

		st.markdown('<div class="chart-gap"></div>', unsafe_allow_html=True)

		c5, c6 = st.columns(2)
		prod_gender = filtered.groupby(["numofproducts", "gender"]).size().reset_index(name="customers")
		fig_prod = px.bar(
			prod_gender,
			x="numofproducts",
			y="customers",
			color="gender",
			barmode="group",
			title="Customers by Product Holders and Gender",
			color_discrete_sequence=chart_colors(2),
		)
		style_figure(fig_prod, theme)
		fig_prod.update_layout(xaxis_title="", legend_title_text="")
		with c5:
			st.plotly_chart(fig_prod, use_container_width=True, config={"displayModeBar": False})

		if "detail_chart_view" not in st.session_state:
			st.session_state.detail_chart_view = "Credit Score Group"
		active_detail_view = st.session_state.detail_chart_view

		if active_detail_view == "Credit Score Group":
			detail_ct = filtered["credit_group_3"].value_counts().sort_index().reset_index()
			detail_ct.columns = ["detail_group", "customers"]
			detail_title = "Customers by Credit Score"
		elif active_detail_view == "Saving Group":
			detail_ct = filtered["saving_group"].value_counts().sort_index().reset_index()
			detail_ct.columns = ["detail_group", "customers"]
			detail_ct["detail_label"] = detail_ct["detail_group"].astype(str).map(_label_with_k)
			detail_ct["detail_label"] = pd.Categorical(
				detail_ct["detail_label"],
				categories=detail_ct["detail_label"].tolist(),
				ordered=True,
			)
			detail_title = "Customers by Saving Group"
		else:
			detail_ct = filtered["estimatedsalary_group"].value_counts().sort_index().reset_index()
			detail_ct.columns = ["detail_group", "customers"]
			detail_ct["detail_label"] = detail_ct["detail_group"].astype(str).map(_label_with_k)
			detail_ct["detail_label"] = pd.Categorical(
				detail_ct["detail_label"],
				categories=detail_ct["detail_label"].tolist(),
				ordered=True,
			)
			detail_title = "Customers by Income Group"

		fig_credit = px.bar(
			detail_ct,
			x="detail_label" if "detail_label" in detail_ct.columns else "detail_group",
			y="customers",
			title=detail_title,
			color_discrete_sequence=chart_colors(1),
		)
		style_figure(fig_credit, theme)
		fig_credit.update_layout(showlegend=False)
		with c6:
			st.plotly_chart(fig_credit, use_container_width=True, config={"displayModeBar": False})
			st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
			render_detail_chart_switcher()

	with right:
		potential_active = filtered[
			(filtered["hascrcard"] == 0)
			& (filtered["exited"] == 0)
			& (filtered["creditscore"] >= 700)
			& (filtered["estimatedsalary"] >= 100000)
			& (filtered["balance"] >= 100000)
		]

		no_saving_customers = filtered[filtered["balance"] == 0]
		potential_non_saving = no_saving_customers[
			(no_saving_customers["creditscore"] >= 700)
			& (no_saving_customers["estimatedsalary"] >= 100000)
			& (no_saving_customers["exited"] == 0)
		]

		saving_customers = filtered[filtered["balance"] > 0]
		potential_saving = saving_customers[
			(saving_customers["balance"] > 100000)
			& (saving_customers["creditscore"] >= 700)
			& (saving_customers["estimatedsalary"] > 100000)
			& (saving_customers["hascrcard"] == 0)
			& (saving_customers["exited"] == 0)
		]

		total = len(filtered)
		pa_pct = (len(potential_active) / total * 100) if total else 0
		pns_pct = (len(potential_non_saving) / total * 100) if total else 0
		ps_pct = (len(potential_saving) / total * 100) if total else 0

		st.markdown(
			f"""
			<div class="findings">
				<h3>Findings &amp; Recommendations</h3>
				<p>Potential active customers (credit card upsell): <strong>{len(potential_active):,} ({pa_pct:.1f}%)</strong></p>
				<p>Strong non-saving profile (savings cross-sell): <strong>{len(potential_non_saving):,} ({pns_pct:.1f}%)</strong></p>
				<p>High-balance, high-credit segment: <strong>{len(potential_saving):,} ({ps_pct:.1f}%)</strong></p>
				<p><strong>Recommendation:</strong> Launch targeted engagement campaigns for inactive members, as inactivity is strongly linked to churn. Cross-sell credit cards to high-balance users to increase product stickiness.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)


def render_churn_analysis_page(df: pd.DataFrame, theme: dict) -> None:
	st.markdown("## Churn Analysis", unsafe_allow_html=True)
	st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

	filter_cols = st.columns([1.5, 1.5, 1.3, 1.3, 1.3, 1.3, 1.3])
	with filter_cols[2]:
		geo_options = sorted(df["geography"].dropna().astype(str).unique().tolist())
		geo_filter = st.selectbox("Select Geography", ["all"] + geo_options, key="churn_geo")
	with filter_cols[3]:
		age_options = ["18-20", "21-23", "24-31", "32-44", "45-60", "above 60"]
		age_filter = st.selectbox("Select Age Group", ["all"] + age_options, key="churn_age")
	with filter_cols[4]:
		tenure_options = sorted(df["tenure"].dropna().astype(int).unique().tolist())
		tenure_filter = st.selectbox(
			"Select Tenure",
			["all"] + [str(value) for value in tenure_options],
			key="churn_tenure",
		)
	with filter_cols[5]:
		product_options = sorted(df["numofproducts"].dropna().astype(int).unique().tolist())
		product_filter = st.selectbox(
			"Select Product Holdings",
			["all"] + [str(value) for value in product_options],
			key="churn_products",
		)
	with filter_cols[6]:
		gender_options = sorted(df["gender"].dropna().astype(str).unique().tolist())
		gender_filter = st.selectbox("Select Gender", ["all"] + gender_options, key="churn_gender")

	filtered = df.copy()
	if geo_filter != "all":
		filtered = filtered[filtered["geography"] == geo_filter]
	if age_filter != "all":
		filtered = filtered[filtered["age_group"].astype(str) == age_filter]
	if tenure_filter != "all":
		filtered = filtered[filtered["tenure"] == int(tenure_filter)]
	if product_filter != "all":
		filtered = filtered[filtered["numofproducts"] == int(product_filter)]
	if gender_filter != "all":
		filtered = filtered[filtered["gender"] == gender_filter]

	strong_engagement_customers = filtered[
		(filtered["creditscore"] >= 700)
		& (filtered["balance"] >= 100000)
		& (filtered["estimatedsalary"] >= 100000)
		& (filtered["isactivemember"] == 1)
	]
	strong_engagement_churn_customers = strong_engagement_customers[strong_engagement_customers["exited"] == 1]
	churn_customers = filtered[filtered["exited"] == 1]

	rate_in_churn = (len(strong_engagement_churn_customers) / len(churn_customers) * 100) if len(churn_customers) else 0
	rate_in_engagement = (
		len(strong_engagement_churn_customers) / len(strong_engagement_customers) * 100
		if len(strong_engagement_customers)
		else 0
	)

	with filter_cols[0]:
		small_metric("Engagement Churn Share (Churned Customers)", f"{rate_in_churn:,.2f}%")
	with filter_cols[1]:
		small_metric("Engagement Churn Rate (Engaged Segment)", f"{rate_in_engagement:,.2f}%")

	st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
	left, right = st.columns([5.2, 1.6], gap="large")

	with left:
		influence_tiers = {
			"Very Strong Influence": ["numofproducts", "age_group", "geography", "hascrcard"],
			"Strong Influence": ["saving_group", "gender", "isactivemember"],
			"Low Influence": ["creditscore_group"],
			"No Influence": ["tenure", "estimatedsalary_group"],
		}
		if "churn_influence_tier" not in st.session_state:
			st.session_state.churn_influence_tier = "Very Strong Influence"

		def _set_churn_tier(selected: str) -> None:
			index_key = f"churn_tier_index_{_slugify(selected)}"
			if st.session_state.get("churn_influence_tier") == selected:
				max_pages = 2 if len(influence_tiers[selected]) > 2 else 1
				st.session_state[index_key] = (st.session_state.get(index_key, 0) + 1) % max_pages
			else:
				st.session_state.churn_influence_tier = selected
				st.session_state[index_key] = 0

		tier_cols = st.columns(4)
		for idx, label in enumerate(influence_tiers.keys()):
			with tier_cols[idx]:
				btn_type = "primary" if st.session_state.churn_influence_tier == label else "secondary"
				# put the small hint inside the button label for readability
				display_label = f"{label}\n(Click again to see more)" if len(influence_tiers[label]) > 2 else label
				st.button(
					display_label,
					use_container_width=True,
					type=btn_type,
					on_click=_set_churn_tier,
					args=(label,),
				)

		selected_tier = st.session_state.churn_influence_tier
		selected_features = influence_tiers[selected_tier]
		index_key = f"churn_tier_index_{_slugify(selected_tier)}"
		page_index = st.session_state.get(index_key, 0)
		start_index = page_index * 2
		features_to_plot = selected_features[start_index : start_index + 2]

		chart_cols = st.columns([1.2, 1.6, 1.6])
		with chart_cols[0]:
			churn_counts = filtered["exited"].value_counts().reindex([0, 1]).fillna(0).reset_index()
			churn_counts.columns = ["status", "customers"]
			churn_counts["status"] = churn_counts["status"].map({0: "Non Churner", 1: "Churner"})
			fig_donut = px.pie(
				churn_counts,
				values="customers",
				names="status",
				title="Churners vs Non Churners",
				hole=0.65,
				color_discrete_sequence=chart_colors(2),
			)
			fig_donut.update_traces(textposition="outside", textinfo="label+percent")
			fig_donut.add_annotation(
				text=f"Total Customers<br><b>{len(filtered):,}</b>",
				x=0.5,
				y=0.5,
				font=dict(size=13, color=theme["text"]),
				showarrow=False,
			)
			style_figure(fig_donut, theme, height=280)
			st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
		churn_only = filtered[filtered["exited"] == 1]
		for col_index in [1, 2]:
			with chart_cols[col_index]:
				if col_index - 1 < len(features_to_plot):
					feature = features_to_plot[col_index - 1]
					ct = churn_only[feature].value_counts().reset_index()
					ct.columns = [feature, "customers"]
					x_col = feature
					if feature in {"saving_group", "estimatedsalary_group"}:
						ct["feature_label"] = ct[feature].astype(str).map(_label_with_k)
						x_col = "feature_label"
					fig = px.bar(
						ct,
						x=x_col,
						y="customers",
						title=f"Churner Count by {feature.replace('_', ' ').title()}",
						color_discrete_sequence=chart_colors(1),
					)
					style_figure(fig, theme, height=280)
					fig.update_layout(showlegend=False, xaxis_title="")
					fig.update_yaxes(showticklabels=True)
					st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
				else:
					st.markdown(
						"""
						<div class="panel" style="min-height:280px; display:flex; align-items:center; justify-content:center;">
						No additional feature for this tier
						</div>
						""",
						unsafe_allow_html=True,
					)

		lower_cols = st.columns([1.4, 1.4])
		with lower_cols[0]:
			if strong_engagement_churn_customers.empty:
				st.info("No strong engagement churn customers in current filters.")
			else:
				age_geo = strong_engagement_churn_customers.groupby(["age_group", "geography"]).size().reset_index(name="customers")
				geo_count = age_geo["geography"].nunique()
				fig_small = px.bar(
					age_geo,
					x="age_group",
					y="customers",
					color="geography",
					barmode="group",
					title="Strong Engagement Churn Customers by Age Group and Geography",
					color_discrete_sequence=chart_colors(geo_count),
				)
				style_figure(fig_small, theme, height=220)
				fig_small.update_layout(legend_title_text="", xaxis_title="")
				fig_small.update_yaxes(showticklabels=True, tickmode="linear", dtick=20)
				st.plotly_chart(fig_small, use_container_width=True, config={"displayModeBar": False})
		with lower_cols[1]:
				# Render hypothesis panel with inline HTML table so it stays inside the panel
				if churn_only.empty:
					st.markdown(
						"""
						<div class="panel" style="min-height:220px; padding:12px;">
							<div style="font-weight:600; margin-bottom:8px;">Hypothesis base on influence churn click</div>
							<div style="padding:14px;">No churn customers available for hypothesis summary.</div>
						</div>
						""",
						unsafe_allow_html=True,
					)
				else:
					hypothesis_rows = []
					for feature in features_to_plot:
						counts = churn_only[feature].value_counts()
						if counts.empty:
							continue
						top_value = counts.index[0]
						display_value = _label_with_k(top_value) if feature in {"saving_group", "estimatedsalary_group"} else str(top_value)
						hypothesis_rows.append(
							{
								"Feature": feature.replace("_", " ").title(),
								"Top Churn Segment": display_value,
								"Churners": int(counts.iloc[0]),
								"Churn Share (%)": round((counts.iloc[0] / len(churn_only)) * 100, 1),
							}
						)
					if hypothesis_rows:
						hypothesis_table = pd.DataFrame(hypothesis_rows)
						html_table = hypothesis_table.to_html(index=False, classes="hypothesis-table", border=0)
						st.markdown(
							f"""
							<div class="panel" style="min-height:220px; padding:12px;">
								<div style="font-weight:600; margin-bottom:8px;">Hypothesis base on influence churn click</div>
								{html_table}
							</div>
							""",
							unsafe_allow_html=True,
						)
					else:
						st.markdown(
							"""
							<div class="panel" style="min-height:220px; padding:12px;">
								<div style="font-weight:600; margin-bottom:8px;">Hypothesis base on influence churn click</div>
								<div style="padding:14px;">No hypothesis summary available for the current tier.</div>
							</div>
							""",
							unsafe_allow_html=True,
						)

		st.markdown("### Strong Engagement Churn Customers", unsafe_allow_html=True)
		cols = [
			"customerid",
			"surname",
			"geography",
			"gender",
			"age",
			"numofproducts",
			"tenure",
			"balance",
			"creditscore",
			"estimatedsalary",
		]
		visible = [col for col in cols if col in strong_engagement_churn_customers.columns]
		table_data = strong_engagement_churn_customers[visible].copy()
		render_theme_dataframe(table_data)

	with right:
		total = len(filtered)
		churn_total = len(churn_customers)
		churn_rate = (churn_total / total * 100) if total else 0
		geo_focus = churn_customers["geography"].mode().iloc[0] if not churn_customers.empty else "n/a"
		age_focus = churn_customers["age_group"].mode().iloc[0] if not churn_customers.empty else "n/a"
		st.markdown(
			f"""
			<div class="findings">
				<h3>Recommendations &amp; Findings</h3>
				<p>Total churn customers: <strong>{churn_total:,} ({churn_rate:.1f}%)</strong></p>
				<p>Top churn geography: <strong>{geo_focus}</strong></p>
				<p>Top churn age group: <strong>{age_focus}</strong></p>
				<p><strong>Finding:</strong> Customers in Germany and the 45-60 age group show the highest statistical likelihood of churning.</p>
				<p><strong>Recommendation:</strong> Develop localized retention plans for the German market and offer customized financial health checkups for the 45-60 age segment.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)


def render_potential_active_page(df: pd.DataFrame, theme: dict) -> None:
	st.markdown("## Potential Active Customers", unsafe_allow_html=True)
	st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

	left, center, right = st.columns([1.2, 5.0, 1.65], gap="large")

	with left:
		filtered = apply_potential_filters(df, key_prefix="potential_active")

	potential_active = filtered[
		(filtered["hascrcard"] == 0)
		& (filtered["exited"] == 0)
		& (filtered["creditscore"] >= 700)
		& (filtered["estimatedsalary"] >= 100000)
		& (filtered["balance"] >= 100000)
	]

	with center:
		k1, k2 = st.columns(2)
		total_filtered = len(filtered)
		potential_total = len(potential_active)
		active_increase_rate = (potential_total / total_filtered * 100) if total_filtered else 0

		with k1:
			small_metric("Total Potential Active Customers", f"{potential_total:,}")
		with k2:
			small_metric("Active Increase Rate", f"{active_increase_rate:,.1f}%")

		st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
		if potential_active.empty:
			st.info("No potential active customers match the current filters.")
		else:
			gender_geo = potential_active.groupby(["geography", "gender"]).size().reset_index(name="customers")
			fig = px.bar(
				gender_geo,
				x="geography",
				y="customers",
				color="gender",
				barmode="group",
				title="Potential Customers by Geography and Gender",
				color_discrete_sequence=chart_colors(2),
			)
			style_figure(fig, theme, height=520)
			fig.update_layout(legend_title_text="", xaxis_title="")
			fig.update_yaxes(showticklabels=True, tickmode="linear", dtick=20)
			st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

	with right:
		total = len(filtered)
		potential_pct = (len(potential_active) / total * 100) if total else 0
		geo_focus = potential_active["geography"].mode().iloc[0] if not potential_active.empty else "n/a"
		gender_focus = potential_active["gender"].mode().iloc[0] if not potential_active.empty else "n/a"
		st.markdown(
			f"""
			<div class="findings">
				<h3>Findings &amp; Recommendations</h3>
				<p>Total potential active customers: <strong>{len(potential_active):,} ({potential_pct:.1f}%)</strong></p>
				<p>Top geography opportunity: <strong>{geo_focus}</strong></p>
				<p>Top gender opportunity: <strong>{gender_focus}</strong></p>
				<p><strong>Finding:</strong> This segment has high income and balances (over 100k) but currently lacks a credit card.</p>
				<p><strong>Recommendation:</strong> Target this specific demographic with premium credit card upsell campaigns, bundling exclusive rewards and onboarding bonuses to increase activation.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)

	st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
	st.markdown("### Potential Customer List", unsafe_allow_html=True)
	cols = [
		"customerid",
		"surname",
		"geography",
		"gender",
		"age",
		"numofproducts",
		"tenure",
		"balance",
		"creditscore",
		"estimatedsalary",
	]
	visible = [col for col in cols if col in potential_active.columns]
	table_data = potential_active[visible].copy()
	render_theme_dataframe(table_data)


def render_strong_engagement_page(df: pd.DataFrame, theme: dict) -> None:
	st.markdown("## Strong Engagement Active Customers", unsafe_allow_html=True)
	st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

	left, center, right = st.columns([1.2, 5.0, 1.65], gap="large")

	with left:
		filtered = apply_potential_filters(df, key_prefix="strong_engagement")

	strong_engagement = filtered[
		(filtered["isactivemember"] == 1)
		& (filtered["exited"] == 0)
		& (filtered["creditscore"] >= 700)
		& (filtered["estimatedsalary"] >= 100000)
		& (filtered["balance"] > 100000)
	]

	with center:
		k1, k2 = st.columns(2)
		total_filtered = len(filtered)
		engaged_total = len(strong_engagement)
		engagement_rate = (engaged_total / total_filtered * 100) if total_filtered else 0

		with k1:
			small_metric("Total Strong Engagement Customers", f"{engaged_total:,}")
		with k2:
			small_metric("Engagement Rate", f"{engagement_rate:,.1f}%")

		st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
		if strong_engagement.empty:
			st.info("No strong engagement customers match the current filters.")
		else:
			age_gender = strong_engagement.groupby(["age_group", "gender"]).size().reset_index(name="customers")
			fig = px.bar(
				age_gender,
				x="age_group",
				y="customers",
				color="gender",
				barmode="group",
				title="Strong Engagement by Age Group and Gender",
				color_discrete_sequence=chart_colors(2),
			)
			style_figure(fig, theme, height=520)
			fig.update_layout(legend_title_text="", xaxis_title="")
			fig.update_yaxes(showticklabels=True, tickmode="linear", dtick=20)
			st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

	with right:
		total = len(filtered)
		engaged_pct = (len(strong_engagement) / total * 100) if total else 0
		geo_focus = strong_engagement["geography"].mode().iloc[0] if not strong_engagement.empty else "n/a"
		age_focus = strong_engagement["age_group"].mode().iloc[0] if not strong_engagement.empty else "n/a"
		st.markdown(
			f"""
			<div class="findings">
				<h3>Findings &amp; Recommendations</h3>
				<p>Total strong engagement customers: <strong>{len(strong_engagement):,} ({engaged_pct:.1f}%)</strong></p>
				<p>Top geography opportunity: <strong>{geo_focus}</strong></p>
				<p>Top age group opportunity: <strong>{age_focus}</strong></p>
				<p><strong>Finding:</strong> These are highly active members with excellent credit scores and substantial balances, making them your most valuable customers.</p>
				<p><strong>Recommendation:</strong> Protect this segment from competitor poaching by offering premium tier banking features, dedicated account managers, and loyalty appreciation rewards.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)

	st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
	st.markdown("### Strong Engagement Customer List", unsafe_allow_html=True)
	cols = [
		"customerid",
		"surname",
		"geography",
		"gender",
		"age",
		"numofproducts",
		"tenure",
		"balance",
		"creditscore",
		"estimatedsalary",
	]
	visible = [col for col in cols if col in strong_engagement.columns]
	table_data = strong_engagement[visible].copy()
	render_theme_dataframe(table_data)


def render_placeholder_page(title: str) -> None:
	st.markdown(f"## {title}", unsafe_allow_html=True)
	st.markdown(
		"<div class='panel' style='padding:16px;'>Content for this page is ready to be wired in.</div>",
		unsafe_allow_html=True,
	)


@st.cache_resource
def get_trained_churn_model(df: pd.DataFrame):
	from sklearn.linear_model import LogisticRegression
	from sklearn.preprocessing import MinMaxScaler
	
	X = df.copy()
	X['gender_scale'] = X['gender'].map({'male': 0, 'female': 1}).fillna(0)
	
	scaler = MinMaxScaler()
	for col in ['numofproducts', 'creditscore', 'age']:
		if col not in X.columns:
			X[col] = 0
	X[['numofproducts_scale', 'creditscore_scale', 'age_scale']] = scaler.fit_transform(X[['numofproducts', 'creditscore', 'age']])
	
	num_dummies = pd.get_dummies(X['numofproducts'], prefix='numofproducts', drop_first=False)
	age_dummies = pd.get_dummies(X['age_group'], prefix='age_group', drop_first=True)
	geo_dummies = pd.get_dummies(X['geography'], prefix='geography', drop_first=True)
	
	X_features = pd.concat([
		num_dummies,
		X[['creditscore_scale']],
		age_dummies,
		X[['gender_scale']],
		geo_dummies,
		X[['saving', 'isactivemember', 'hascrcard']]
	], axis=1)

	cols_expected = [
		'numofproducts_1', 'numofproducts_2', 'numofproducts_3', 'numofproducts_4',
		'creditscore_scale', 
		'age_group_21-23', 'age_group_24-31', 'age_group_32-44', 'age_group_45-60', 'age_group_above 60',
		'gender_scale', 'geography_germany', 'geography_spain',
		'saving', 'isactivemember', 'hascrcard'
	]
	
	X_model = X_features.reindex(columns=cols_expected, fill_value=0).astype(float)
	y = X['exited']

	model = LogisticRegression(random_state=42, max_iter=1000)
	model.fit(X_model, y)

	return model, scaler, cols_expected


def render_churn_prediction_page(df: pd.DataFrame, theme: dict) -> None:
	import plotly.graph_objects as go
	
	st.markdown("## Customer Churn Simulator", unsafe_allow_html=True)
	st.markdown(
		"<p style='margin-top:-10px; color:var(--text-muted); font-size:14px;'>"
		"Adjust customer profiles in real-time to predict churn risk using our trained Logistic Regression model.</p>",
		unsafe_allow_html=True,
	)
	st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

	model, scaler, feature_cols = get_trained_churn_model(df)

	left, right = st.columns([1.5, 2.5], gap="large")

	with left:
		st.markdown("### Customer Profile", unsafe_allow_html=True)
		with st.container(border=True):
			st.markdown("#### Demographics", unsafe_allow_html=True)
			age = st.slider("Age", min_value=18, max_value=100, value=42)
			gender = st.selectbox("Gender", ["Female", "Male"])
			geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
			
			st.markdown("#### Financials", unsafe_allow_html=True)
			credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
			balance = st.number_input("Balance (€)", min_value=0, max_value=300000, value=50000, step=1000)
			salary = st.number_input("Estimated Salary (€)", min_value=0, max_value=300000, value=100000, step=1000)
			
			st.markdown("#### Engagement", unsafe_allow_html=True)
			products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
			has_card = st.toggle("Has Credit Card", value=True)
			is_active = st.toggle("Active Member", value=True)

	with right:
		st.markdown("### Prediction & Insights", unsafe_allow_html=True)
		
		input_df = pd.DataFrame([{
			'age': age,
			'gender': gender.lower(),
			'geography': geography.lower(),
			'creditscore': credit_score,
			'balance': balance,
			'estimatedsalary': salary,
			'numofproducts': products,
			'hascrcard': 1 if has_card else 0,
			'isactivemember': 1 if is_active else 0,
		}])
		
		input_df['saving'] = (input_df['balance'] > 0).astype(int)
		input_df['gender_scale'] = input_df['gender'].map({'male': 0, 'female': 1})
		
		scaled_vals = scaler.transform(input_df[['numofproducts', 'creditscore', 'age']])
		input_df['numofproducts_scale'] = scaled_vals[:, 0]
		input_df['creditscore_scale'] = scaled_vals[:, 1]
		input_df['age_scale'] = scaled_vals[:, 2]
		
		if age <= 20: age_bin = "18-20"
		elif age <= 23: age_bin = "21-23"
		elif age <= 31: age_bin = "24-31"
		elif age <= 44: age_bin = "32-44"
		elif age <= 60: age_bin = "45-60"
		else: age_bin = "above 60"
		
		input_vector = {col: 0.0 for col in feature_cols}
			
		input_vector[f'numofproducts_{products}'] = 1.0
		input_vector['creditscore_scale'] = input_df['creditscore_scale'].iloc[0]
		if f'age_group_{age_bin}' in feature_cols:
			input_vector[f'age_group_{age_bin}'] = 1.0
		input_vector['gender_scale'] = input_df['gender_scale'].iloc[0]
		if f'geography_{geography.lower()}' in feature_cols:
			input_vector[f'geography_{geography.lower()}'] = 1.0
		input_vector['saving'] = input_df['saving'].iloc[0]
		input_vector['isactivemember'] = input_df['isactivemember'].iloc[0]
		input_vector['hascrcard'] = input_df['hascrcard'].iloc[0]
		
		X_pred = pd.DataFrame([input_vector], columns=feature_cols)
		
		prob = model.predict_proba(X_pred)[0][1]
		risk_label = "High Risk of Churn" if prob >= 0.61 else "Likely to Stay"
		risk_color = "#ef4444" if prob >= 0.61 else "#22c55e"
		
		fig_gauge = go.Figure(go.Indicator(
			mode="gauge+number",
			value=prob * 100,
			number={'valueformat': '.1f', 'suffix': "%", 'font': {'color': theme['text'], 'size': 40}},
			gauge={
				'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': theme['axis'], 'ticksuffix': '%'},
				'bar': {'color': risk_color},
				'bgcolor': theme['chart_bg'],
				'borderwidth': 2,
				'bordercolor': theme['border'],
				'steps': [
					{'range': [0, 61], 'color': theme['surface']},
					{'range': [61, 100], 'color': theme['grid']}
				],
				'threshold': {
					'line': {'color': "#ef4444", 'width': 4},
					'thickness': 0.75,
					'value': 61
				}
			}
		))
		style_figure(fig_gauge, theme, height=280)
		fig_gauge.update_layout(
			title_text="Churn Probability",
			title_x=0.0,
			margin=dict(l=20, r=20, t=60, b=20)
		)
		fig_gauge.add_annotation(
			x=1.0, y=1.2,
			xref="paper", yref="paper",
			text="<b style='color:#ef4444; font-size:18px; vertical-align:middle;'>▬</b> <span style='vertical-align:middle;'>High Risk Threshold</span>",
			showarrow=False,
			xanchor="right",
			yanchor="top",
			font=dict(size=14, color=theme['text_muted'])
		)
		st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
		
		st.markdown(f"<div style='text-align:center; font-size:22px; font-weight:700; color:{risk_color}; margin-top:-20px; margin-bottom:20px;'>{risk_label}</div>", unsafe_allow_html=True)
		
		impacts = model.coef_[0] * X_pred.iloc[0].values
		impact_df = pd.DataFrame({
			'Feature': feature_cols,
			'Impact': impacts
		})
		
		def _clean_name(name):
			name = name.replace('numofproducts_', 'Products: ')
			name = name.replace('age_group_', 'Age: ')
			name = name.replace('geography_', 'Geo: ')
			name = name.replace('_scale', '')
			name = name.replace('hascrcard', 'Has Credit Card')
			name = name.replace('isactivemember', 'Active Member')
			return name.title()
			
		impact_df['Feature'] = impact_df['Feature'].apply(_clean_name)
		impact_df = impact_df[impact_df['Impact'] != 0].copy()
		impact_df['AbsImpact'] = impact_df['Impact'].abs()
		impact_df = impact_df.sort_values('AbsImpact', ascending=True).tail(8)
		
		impact_df['Impact Type'] = impact_df['Impact'].apply(lambda x: "Increases Risk" if x > 0 else "Lowers Risk")
		
		fig_bar = px.bar(
			impact_df,
			x='Impact',
			y='Feature',
			orientation='h',
			color='Impact Type',
			color_discrete_map={"Increases Risk": "#ef4444", "Lowers Risk": "#22c55e"},
			title="Top Factors Influencing This Prediction",
		)
		fig_bar.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': impact_df['Feature']})
		style_figure(fig_bar, theme, height=320)
		fig_bar.update_layout(xaxis_title="Impact Score", yaxis_title="")
		fig_bar.update_yaxes(showticklabels=True)
		st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
	if "theme_mode" not in st.session_state:
		st.session_state.theme_mode = "Dark"
	theme_mode = st.session_state.theme_mode
	theme = get_theme(theme_mode)
	style_page(theme)

	with st.container():
		render_navbar(theme_mode)

	df = load_data()

	if st.session_state.current_page == "overview":
		render_overview(df, theme)
	elif st.session_state.current_page == "potential_active":
		render_potential_active_page(df, theme)
	elif st.session_state.current_page == "churn_analysis":
		render_churn_analysis_page(df, theme)
	elif st.session_state.current_page == "churn_prediction":
		render_churn_prediction_page(df, theme)
	else:
		render_strong_engagement_page(df, theme)


if __name__ == "__main__":
	main()
