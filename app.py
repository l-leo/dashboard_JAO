import datetime as dt
from typing import List, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

JAO_BASE_URL = "https://publicationtool.jao.eu/core/api/data"
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"

NETPOS_CHUNK_DAYS = 2
SHADOW_CHUNK_DAYS = 1

DEFAULT_START = dt.date(2026, 1, 1)
DEFAULT_END = dt.date(2026, 1, 2)
DEFAULT_ENTSOE_KEY = "87f0225e-afc7-489d-b36e-0702d9e0a915"

EIC_CODES = {
    "AT": "10YAT-APG------L",
    "DE_LU": "10Y1001A1001A82H",
    "CZ": "10YCZ-CEPS-----N",
    "HU": "10YHU-MAVIR----U",
    "SI": "10YSI-ELES-----O",
    "SK": "10YSK-SEPS-----K",
}


@st.cache_data(ttl=3600)
def fetch_jao_endpoint(endpoint: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    url = f"{JAO_BASE_URL}/{endpoint}"
    params = {
        "FromUtc": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "ToUtc": end.strftime("%Y-%m-%dT%H:%M:%S.999Z"),
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    return pd.DataFrame(data)


def chunked_date_ranges(start: dt.date, end: dt.date, chunk_days: int) -> List[Tuple[dt.datetime, dt.datetime]]:
    ranges = []
    current = start
    while current <= end:
        chunk_end = min(current + dt.timedelta(days=chunk_days - 1), end)
        start_dt = dt.datetime.combine(current, dt.time.min)
        end_dt = dt.datetime.combine(chunk_end, dt.time.max)
        ranges.append((start_dt, end_dt))
        current = chunk_end + dt.timedelta(days=1)
    return ranges


def load_jao_data(endpoint: str, start: dt.date, end: dt.date, chunk_days: int) -> pd.DataFrame:
    frames = []
    for chunk_start, chunk_end in chunked_date_ranges(start, end, chunk_days):
        try:
            frame = fetch_jao_endpoint(endpoint, chunk_start, chunk_end)
            frames.append(frame)
        except requests.RequestException as exc:
            st.warning(f"JAO API Fehler für {endpoint}: {exc}")
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def transform_net_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={"dateTimeUtc": "datetime"})
    hub_columns = [col for col in df.columns if col.startswith("hub_")]
    if not hub_columns:
        return pd.DataFrame()
    melted = df.melt(
        id_vars=["datetime"],
        value_vars=hub_columns,
        var_name="hub",
        value_name="net_position",
    )
    melted["hub"] = melted["hub"].str.replace("hub_", "", regex=False)
    melted["datetime"] = pd.to_datetime(melted["datetime"], utc=True)
    return melted


def normalize_shadow_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    lower_map = {col.lower(): col for col in df.columns}
    rename_map = {}
    for key in ["datetimeutc", "cnecname", "shadowprice", "tso", "cne_tso"]:
        if key in lower_map:
            rename_map[lower_map[key]] = key
    df = df.rename(columns=rename_map)
    if "datetimeutc" in df.columns:
        df = df.rename(columns={"datetimeutc": "datetime"})
    if "cnecname" in df.columns:
        df = df.rename(columns={"cnecname": "cnec_name"})
    if "shadowprice" in df.columns:
        df = df.rename(columns={"shadowprice": "shadow_price"})
    if "cne_tso" in df.columns:
        df = df.rename(columns={"cne_tso": "tso"})
    df["datetime"] = pd.to_datetime(df.get("datetime"), utc=True, errors="coerce")
    df["shadow_price"] = pd.to_numeric(df.get("shadow_price"), errors="coerce")
    if "cnec_name" not in df.columns:
        df["cnec_name"] = ""
    if "tso" not in df.columns:
        df["tso"] = ""

    tso_values = df["tso"].fillna("")
    cnec_values = df["cnec_name"].fillna("")
    mask = tso_values.str.contains("APG", case=False) | cnec_values.str.contains(
        "APG|_AT_", case=False, regex=True
    )
    return df.loc[mask].copy()


@st.cache_data(ttl=3600)
def fetch_entsoe_prices(
    security_token: str, start: dt.date, end: dt.date, countries: List[str]
) -> pd.DataFrame:
    frames = []
    namespace = {
        "ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"
    }
    period_start = dt.datetime.combine(start, dt.time.min).strftime("%Y%m%d%H%M")
    period_end = dt.datetime.combine(end + dt.timedelta(days=1), dt.time.min).strftime(
        "%Y%m%d%H%M"
    )
    for country in countries:
        eic = EIC_CODES.get(country)
        if not eic:
            continue
        params = {
            "securityToken": security_token,
            "documentType": "A44",
            "in_Domain": eic,
            "out_Domain": eic,
            "periodStart": period_start,
            "periodEnd": period_end,
        }
        try:
            response = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            st.warning(f"ENTSO-E API Fehler für {country}: {exc}")
            continue
        root = ET.fromstring(response.text)
        for timeseries in root.findall("ns:TimeSeries", namespace):
            period = timeseries.find("ns:Period", namespace)
            if period is None:
                continue
            resolution = period.findtext("ns:resolution", default="", namespaces=namespace)
            start_text = period.findtext("ns:timeInterval/ns:start", namespaces=namespace)
            if not start_text:
                continue
            start_dt = dt.datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            points = period.findall("ns:Point", namespace)
            for point in points:
                position_text = point.findtext("ns:position", namespaces=namespace)
                price_text = point.findtext("ns:price.amount", namespaces=namespace)
                if not position_text or not price_text:
                    continue
                position = int(position_text)
                if resolution == "PT15M":
                    delta = dt.timedelta(minutes=15 * (position - 1))
                else:
                    delta = dt.timedelta(hours=position - 1)
                frames.append(
                    {
                        "datetime": start_dt + delta,
                        "country": country,
                        "price": float(price_text),
                    }
                )
    if not frames:
        return pd.DataFrame()
    prices = pd.DataFrame(frames)
    prices["datetime"] = pd.to_datetime(prices["datetime"], utc=True)
    prices = prices.drop_duplicates(subset=["datetime", "country"])
    return prices


st.set_page_config(page_title="JAO Dashboard AT", layout="wide")

st.title("JAO Dashboard für Österreich")

with st.sidebar:
    st.header("Einstellungen")
    max_date = dt.date.today() - dt.timedelta(days=2)
    start_date = st.date_input(
        "Von", value=st.session_state.get("start", DEFAULT_START), max_value=max_date
    )
    end_date = st.date_input(
        "Bis", value=st.session_state.get("end", DEFAULT_END), max_value=max_date
    )
    entsoe_key = st.text_input(
        "ENTSO-E API-Key",
        type="password",
        value=st.session_state.get("entsoe_key", DEFAULT_ENTSOE_KEY),
    )

    try:
        jao_hub_options = fetch_available_jao_hubs()
    except requests.RequestException as exc:
        st.warning(f"JAO-Hub-Liste konnte nicht geladen werden: {exc}")
        jao_hub_options = ["AT", "DE", "CZ", "HU", "SI", "SK"]

    default_hubs = st.session_state.get("jao_hubs", ["AT", "DE"])
    default_hubs = [hub for hub in default_hubs if hub in jao_hub_options]
    if not default_hubs and jao_hub_options:
        default_hubs = jao_hub_options[:2]

    jao_hubs = st.multiselect(
        "JAO-Länder/Hubs (Nettoposition)",
        options=jao_hub_options,
        default=default_hubs,
    )

    countries = st.multiselect(
        "Preis-Länder (ENTSO-E)",
    countries = st.multiselect(
        "Länder",
        options=list(EIC_CODES.keys()),
        default=st.session_state.get("countries", ["AT", "DE_LU"]),
    )
    load_clicked = st.button("Daten laden")

if load_clicked:
    st.session_state.load_data = True
    st.session_state.start = start_date
    st.session_state.end = end_date
    st.session_state.countries = countries
    st.session_state.entsoe_key = entsoe_key

if start_date > end_date:
    st.error("Das Startdatum muss vor dem Enddatum liegen.")
    st.stop()

if (end_date - start_date).days > 30:
    st.warning("Große Zeiträume können zu langen Ladezeiten führen.")

if st.session_state.get("load_data"):
    with st.spinner("Lade Daten..."):
        netpos_raw = load_jao_data("netPos", start_date, end_date, NETPOS_CHUNK_DAYS)
        shadow_raw = load_jao_data("shadowPrices", start_date, end_date, SHADOW_CHUNK_DAYS)
        netpos = transform_net_positions(netpos_raw)
        shadow = normalize_shadow_prices(shadow_raw)
        prices = fetch_entsoe_prices(entsoe_key, start_date, end_date, countries)
    st.session_state.netpos = netpos
    st.session_state.shadow = shadow
    st.session_state.prices = prices

netpos = st.session_state.get("netpos", pd.DataFrame())
shadow = st.session_state.get("shadow", pd.DataFrame())
prices = st.session_state.get("prices", pd.DataFrame())

tab_overview, tab_cnec, tab_prices, tab_export = st.tabs(
    ["Übersicht", "Engpässe (CNECs)", "Preise", "Export"]
)

with tab_overview:
    st.subheader("Nettoposition Österreich")
    if netpos.empty:
        st.info("Bitte Daten laden, um die Übersicht zu sehen.")
    else:
        at_net = netpos[netpos["hub"] == "AT"].copy()
        if not at_net.empty:
            avg_val = at_net["net_position"].mean()
            max_val = at_net["net_position"].max()
            min_val = at_net["net_position"].min()
            export_hours = (at_net["net_position"] > 0).sum()
        else:
            avg_val = max_val = min_val = export_hours = 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ø Nettoposition (MW)", f"{avg_val:,.1f}")
        col2.metric("Maximum (MW)", f"{max_val:,.1f}")
        col3.metric("Minimum (MW)", f"{min_val:,.1f}")
        col4.metric("Export-Stunden", f"{export_hours}")

        if not at_net.empty:
            chart_data = at_net.set_index("datetime")["net_position"]
            st.line_chart(chart_data)

        avg_hubs = (
            netpos.groupby("hub")["net_position"].mean().sort_values(ascending=False)
        )
        st.subheader("Durchschnittliche Nettopositionen")
        st.dataframe(avg_hubs.reset_index().rename(columns={"net_position": "MW"}))

with tab_cnec:
    st.subheader("CNEC Engpässe")
    with st.expander("Glossar"):
        st.markdown(
            """
            - **CNEC**: Critical Network Element with Contingency
            - **Shadow Price**: Grenzkosten der Kapazität (€/MWh)
            - **Bindend**: Shadow Price > 0
            """
        )

    if shadow.empty:
        st.info("Bitte Daten laden, um Engpässe zu analysieren.")
    else:
        shadow["binding"] = shadow["shadow_price"] > 0
        binding = shadow[shadow["binding"]]
        col1, col2, col3 = st.columns(3)
        col1.metric("Bindende Stunden", f"{binding.shape[0]}")
        col2.metric(
            "Ø Shadow Price (bindend)",
            f"{binding['shadow_price'].mean():,.2f}" if not binding.empty else "0",
        )
        col3.metric(
            "Max Shadow Price", f"{shadow['shadow_price'].max():,.2f}" if not shadow.empty else "0"
        )

        top_cnecs = (
            binding.groupby("cnec_name")
            .agg(
                sum_eur=("shadow_price", "sum"),
                avg_eur=("shadow_price", "mean"),
                hours=("binding", "sum"),
            )
            .sort_values("sum_eur", ascending=False)
            .head(10)
            .reset_index()
        )
        st.subheader("Top 10 Engpässe")
        st.dataframe(top_cnecs)
        if not top_cnecs.empty:
            st.bar_chart(top_cnecs.set_index("cnec_name")["sum_eur"])

        shadow_dates = shadow["datetime"].dt.date.dropna().unique().tolist()
        if shadow_dates:
            selected_date = st.selectbox("Datum", options=shadow_dates)
            selected_hour = st.selectbox(
                "Stunde", options=[f"{hour:02d}:00" for hour in range(24)]
            )
            hour_value = int(selected_hour.split(":")[0])
            filtered = shadow[
                (shadow["datetime"].dt.date == selected_date)
                & (shadow["datetime"].dt.hour == hour_value)
                & (shadow["shadow_price"] > 0)
            ]
            display_cols = ["datetime", "cnec_name", "shadow_price", "tso"]
            st.dataframe(filtered[display_cols])

with tab_prices:
    st.subheader("Day-Ahead Preise")
    if prices.empty:
        st.info("Bitte Daten laden, um Preise zu sehen.")
    else:
        at_prices = prices[prices["country"] == "AT"]
        col1, col2, col3 = st.columns(3)
        if at_prices.empty:
            col1.metric("Ø Preis (€/MWh)", "0")
            col2.metric("Minimum (€/MWh)", "0")
            col3.metric("Maximum (€/MWh)", "0")
        else:
            col1.metric("Ø Preis (€/MWh)", f"{at_prices['price'].mean():,.2f}")
            col2.metric("Minimum (€/MWh)", f"{at_prices['price'].min():,.2f}")
            col3.metric("Maximum (€/MWh)", f"{at_prices['price'].max():,.2f}")

        pivot = prices.pivot_table(index="datetime", columns="country", values="price")
        st.line_chart(pivot)

        if "AT" in pivot.columns:
            spread = pivot.sub(pivot["AT"], axis=0)
            spread = spread.drop(columns=["AT"], errors="ignore")
            st.subheader("Preisspreads zu Österreich")
            st.line_chart(spread)

with tab_export:
    st.subheader("Datenexport")
    datasets = {
        "Net Positions": netpos,
        "Shadow Prices": shadow,
        "ENTSO-E Preise": prices,
    }
    selection = st.selectbox("Datensatz", options=list(datasets.keys()))
    selected_df = datasets[selection]
    st.dataframe(selected_df.head(100))
    if not selected_df.empty:
        csv = selected_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "CSV herunterladen", data=csv, file_name=f"{selection}.csv", mime="text/csv"
        )
