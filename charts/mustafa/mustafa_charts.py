import pandas as pd
import plotly.express as px
import streamlit as st

PLOT_TEMPLATE = "plotly_white"

def render_violin_plot(df: pd.DataFrame) -> None:
    st.subheader("Violin Plot")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    short_cat_cols = [c for c in category_cols if df[c].nunique() < 20]

    if not numeric_cols:
        st.warning("No numeric columns found for violin plot.")
        return

    with st.expander("Graphic Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            y_axis = st.selectbox("Y axis:", options=numeric_cols, index=0)
        with col2:
            group_col = st.selectbox("X axis (Optional):", options=["None"] + short_cat_cols)
            use_color = st.checkbox("Use Colors", value=True)

    x_val = group_col if group_col != "None" else None
    color_val = group_col if (group_col != "None" and use_color) else None

    fig = px.violin(
        df, y=y_axis, x=x_val, color=color_val,
        box=True, points="all", template=PLOT_TEMPLATE,
        title=f"{y_axis} Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_plot(df: pd.DataFrame) -> None:
    st.subheader("3D Scatter Plot")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    short_cat_cols = [c for c in category_cols if df[c].nunique() < 30]

    if len(numeric_cols) < 3:
        st.warning("At least 3 numeric columns are required for a 3D scatter plot.")
        return

    with st.expander("Graphic Settings", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_axis = st.selectbox("X axis:", options=numeric_cols, index=0)
        with col2:
            y_axis = st.selectbox("Y axis:", options=numeric_cols, index=1)
        with col3:
            z_axis = st.selectbox("Z axis:", options=numeric_cols, index=2)
        with col4:
            color_col = st.selectbox("Color by:", options=["None"] + short_cat_cols)

    color_val = color_col if color_col != "None" else None

    fig = px.scatter_3d(
        df,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color=color_val,
        template=PLOT_TEMPLATE,
        title=f"{x_axis} vs {y_axis} vs {z_axis}",
        hover_data=df.columns.tolist()[:5],
        height=700
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            zaxis_title=z_axis
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)



def render_treemap(df: pd.DataFrame) -> None:
    st.subheader("Treemap")

    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    valid_cats = [c for c in category_cols if df[c].nunique() > 1]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not valid_cats:
        st.warning("No valid data found for treemap.")
        return

    with st.expander("Graphic Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            path_col = st.selectbox("Category:", options=valid_cats, index=0)
            color_mode = st.radio("Coloring:", ["By Category", "By Value (Heatmap)"], horizontal=True)
        with col2:
            opts = ["Row Count"] + numeric_cols
            value_col = st.selectbox("Box Size:", options=opts)

    plot_df = df.copy()
    
    if value_col == "Row Count":
        plot_df["_count"] = 1
        values_arg = "_count"
    else:
        values_arg = value_col

    if "Value" in color_mode:
        color_arg = values_arg
        color_scale = "Viridis"
    else:
        color_arg = path_col
        color_scale = None

    fig = px.treemap(
        plot_df, path=[path_col], values=values_arg,
        color=color_arg, color_continuous_scale=color_scale,
        template=PLOT_TEMPLATE, title=f"{path_col}-based Distribution"
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Value"))
    st.plotly_chart(fig, use_container_width=True)
