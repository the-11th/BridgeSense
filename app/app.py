"""
Streamlit application for predicting bridge condition ratings.

This version provides dropdown menus for categorical variables and
descriptive labels with units for numeric fields.  When fields are
left blank, median imputation and missing flags are handled in
processing.py.
"""

import streamlit as st
import joblib

# Import transform_data from the processing module inside the `app` package.
# This ensures we use the correct preprocessing logic that loads
# column definitions from app/columns.joblib and avoids ambiguity.
from processing import transform_data  # type: ignore


@st.cache_resource
def load_model_and_columns():
    """Load the trained model and column names from joblib files in the app directory.

    The model and its associated column definitions reside in the
    `app` subdirectory.  Columns may be stored as a pandas Index or other
    iterable; converting them to a list avoids ambiguity when used in
    boolean expressions or set operations.
    """
    model = joblib.load("bridge_risk_model.joblib")
    columns_raw = joblib.load("columns.joblib")
    try:
        columns = list(columns_raw)
    except TypeError:
        columns = columns_raw
    return model, columns


model, model_columns = load_model_and_columns()
label_map = {0: "Poor", 1: "Fair", 2: "Good"}


def build_input_form():
    """Render the user input form with units and category options."""
    st.sidebar.header("Bridge Details")
    st.sidebar.markdown(
        "Fill in the details below. Units are noted in parentheses; "
        "leave fields blank to use training medians or trigger a missing flag."
    )

    form = st.sidebar.form("bridge_form")

    # Year Built (numeric year)
    year_built = form.text_input(
        "27 - Year Built (year)",
        placeholder="e.g. 1985"
    )
    # Average Daily Traffic (vehicles)
    adt = form.text_input(
        "29 - Average Daily Traffic (vehicles/day)",
        placeholder="Enter number"
    )
    # Main Span Material (category)
    # Allow a blank selection for categorical options by prepending an empty
    # string.  Streamlit selectbox requires a default selection, so the
    # empty string acts as “no selection” and will trigger median
    # imputation for categorical variables (handled in processing.py via
    # one-hot encoding and column alignment).
    material_options = [
        "",  # blank option
        "Concrete",
        "Steel",
        "Prestressed Concrete",
        "Timber",
        "Masonry",
        "Aluminum",
        "Other",
    ]
    main_span_material = form.selectbox(
        "43A - Main Span Material", material_options, index=0
    )
    # Main Span Design (category)
    design_options = [
        "",  # blank option
        "Girder",
        "Truss - Through",
        "Truss - Deck",
        "Slab",
        "Arch",
        "Suspension",
        "Cable-Stayed",
        "Culvert",
        "Other",
    ]
    main_span_design = form.selectbox(
        "43B - Main Span Design", design_options, index=0
    )
    # Number of Spans in Main Unit
    num_spans = form.text_input(
        "45 - Number of Spans in Main Unit",
        placeholder="Enter number"
    )
    # Structure Length (feet)
    structure_length = form.text_input(
        "49 - Structure Length (ft.)",
        placeholder="Enter length in feet"
    )
    # Bridge Age (years)
    bridge_age = form.text_input(
        "Bridge Age (yr)",
        placeholder="Enter age in years"
    )
    # Deck Area (square feet)
    deck_area = form.text_input(
        "CAT29 - Deck Area (sq. ft.)",
        placeholder="Enter area in sq. ft."
    )
    # Year Reconstructed
    year_reconstructed = form.text_input(
        "106 - Year Reconstructed (year)",
        placeholder="Enter year"
    )
    # Skew Angle (degrees)
    skew_angle = form.text_input(
        "34 - Skew Angle (degrees)",
        placeholder="Enter degrees"
    )
    # Length of Maximum Span (feet)
    max_span_length = form.text_input(
        "48 - Length of Maximum Span (ft.)",
        placeholder="Enter length in feet"
    )
    # Bridge Roadway Width Curb to Curb (feet)
    roadway_width = form.text_input(
        "51 - Bridge Roadway Width Curb to Curb (ft.)",
        placeholder="Enter width in feet"
    )
    # Designated Inspection Frequency (months or cycles)
    inspection_freq = form.text_input(
        "91 - Designated Inspection Frequency (months)",
        placeholder="Enter months"
    )
    # Operating Rating (US tons)
    operating_rating = form.text_input(
        "64 - Operating Rating (US tons)",
        placeholder="Enter tons"
    )
    # Inventory Rating (US tons)
    inventory_rating = form.text_input(
        "66 - Inventory Rating (US tons)",
        placeholder="Enter tons"
    )
    # Year of Average Daily Traffic
    year_adt = form.text_input(
        "30 - Year of Average Daily Traffic (year)",
        placeholder="Enter year"
    )
    # Average Daily Truck Traffic (Percent ADT)
    pct_truck = form.text_input(
        "109 - Average Daily Truck Traffic (Percent of ADT)",
        placeholder="Enter percent"
    )
    # Future Average Daily Traffic
    future_adt = form.text_input(
        "114 - Future Average Daily Traffic (vehicles/day)",
        placeholder="Enter number"
    )
    # Year of Future Average Daily Traffic
    year_future_adt = form.text_input(
        "115 - Year of Future Average Daily Traffic (year)",
        placeholder="Enter year"
    )
    # Total Project Cost (currency unspecified)
    project_cost = form.text_input(
        "96 - Total Project Cost (USD)",
        placeholder="Enter cost"
    )
    # Computed Average Daily Truck Traffic (Volume)
    computed_truck = form.text_input(
        "Computed - Average Daily Truck Traffic (Volume)",
        placeholder="Enter number"
    )
    # Average Relative Humidity (percent)
    avg_rel_humidity = form.text_input(
        "Average Relative Humidity (%)",
        placeholder="Enter percent"
    )
    # Average Temperature (°F)
    avg_temp = form.text_input(
        "Average Temperature (°F)",
        placeholder="Enter temperature"
    )
    # Maximum Temperature (°F)
    max_temp = form.text_input(
        "Maximum Temperature (°F)",
        placeholder="Enter temperature"
    )
    # Minimum Temperature (°F)
    min_temp = form.text_input(
        "Minimum Temperature (°F)",
        placeholder="Enter temperature"
    )
    # Mean Wind Speed (mph)
    mean_wind_speed = form.text_input(
        "Mean Wind Speed (mph)",
        placeholder="Enter speed"
    )
    # Was Reconstructed (yes/no) with blank option
    was_reconstructed_options = ["", "No", "Yes"]
    was_reconstructed = form.selectbox(
        "Was_Reconstructed", was_reconstructed_options, index=0
    )

    submit = form.form_submit_button("Predict")

    if submit:
        # Helper to convert string inputs to floats if possible
        def to_float_or_none(val: str):
            if val in (None, ""):
                return None
            try:
                return float(val)
            except ValueError:
                return None

        return {
            "27 - Year Built": to_float_or_none(year_built),
            "29 - Average Daily Traffic": to_float_or_none(adt),
            "43A - Main Span Material": main_span_material,
            "43B - Main Span Design": main_span_design,
            "45 - Number of Spans in Main Unit": to_float_or_none(num_spans),
            "49 - Structure Length (ft.)": to_float_or_none(structure_length),
            "Bridge Age (yr)": to_float_or_none(bridge_age),
            "CAT29 - Deck Area (sq. ft.)": to_float_or_none(deck_area),
            "106 - Year Reconstructed": to_float_or_none(year_reconstructed),
            "34 - Skew Angle (degrees)": to_float_or_none(skew_angle),
            "48 - Length of Maximum Span (ft.)": to_float_or_none(max_span_length),
            "51 - Bridge Roadway Width Curb to Curb (ft.)": to_float_or_none(roadway_width),
            "91 - Designated Inspection Frequency": to_float_or_none(inspection_freq),
            "64 - Operating Rating (US tons)": to_float_or_none(operating_rating),
            "66 - Inventory Rating (US tons)": to_float_or_none(inventory_rating),
            "30 - Year of Average Daily Traffic": to_float_or_none(year_adt),
            "109 - Average Daily Truck Traffic (Percent ADT)": to_float_or_none(pct_truck),
            "114 - Future Average Daily Traffic": to_float_or_none(future_adt),
            "115 - Year of Future Average Daily Traffic": to_float_or_none(year_future_adt),
            "96 - Total Project Cost": to_float_or_none(project_cost),
            "Computed - Average Daily Truck Traffic (Volume)": to_float_or_none(computed_truck),
            "Average Relative Humidity": to_float_or_none(avg_rel_humidity),
            "Average Temperature": to_float_or_none(avg_temp),
            "Maximum Temperature": to_float_or_none(max_temp),
            "Minimum Temperature": to_float_or_none(min_temp),
            "Mean Wind Speed": to_float_or_none(mean_wind_speed),
            "Was_Reconstructed": was_reconstructed,
        }

    return {}


def main():
    """Main function to run the Streamlit app."""
    st.title("BridgeSense Condition Prediction")
    # Display the project overview and training results at the top of
    # the page.  This ensures that the purpose of the project and key
    # training insights are visible even before the form is filled out.
    show_training_overview()

    user_input = build_input_form()
    if not user_input:
        # If the user hasn't submitted the form, return early.  The
        # overview has already been displayed above.
        return
    try:
        df_transformed = transform_data(user_input)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        return

    # Align columns to model
    missing_cols = set(model_columns) - set(df_transformed.columns)
    for col in missing_cols:
        df_transformed[col] = 0
    df_transformed = df_transformed[model_columns]

    try:
        prediction = model.predict(df_transformed)[0]
        probas = model.predict_proba(df_transformed)[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    predicted_label = label_map.get(int(prediction), "Unknown")
    proba_dict = {label_map[i]: round(float(probas[i]), 3) for i in range(len(probas))}

    st.write("### Prediction Result")
    st.success(f"The predicted condition is: **{predicted_label}**")
    st.write("### Class Probabilities")
    st.table(proba_dict)

    # The overview is shown at the top of the page, so we don't need
    # to show it again here.


def show_training_overview():
    """Display the project purpose and training results with images."""
    st.markdown("---")
    st.header("Project Purpose and Training Results")
    st.write(
        """
        **Project Objective**
        This project's goal is to predict if a bridge in Georgia will receive 
        a **Poor**, **Fair**, or **Good** condition rating using available NBI 
        features such as traffic, environmental factors, and construction material. 
        Given that a poor rating indicates serious problems that may lead to 
        failure in the near future, this prediction provides critical insight 
        that can be used proactively to take active measures and avoid collapse, 
        rather than waiting for the scheduled times where bridges are inspected.

        **Key Training Insights**

        To train the model, we used features describing traffic,
        structural dimensions, age, and environmental conditions.  An
        XGBoost classifier was chosen for its strong performance on
        imbalanced data.  We also engineered interaction features, such
        as the product of bridge age and truck traffic volume, to
        capture nonlinear relationships.
        """
    )
    # Show the boxplot illustrating the distribution of bridge age by
    # condition.  Older bridges tend to receive poorer ratings, but
    # overlap exists across all classes.
    st.image(
        "age_condition_boxplot.png",
        caption="Distribution of bridge age grouped by condition rating",
        use_container_width=True,
    )
    # Show SHAP feature importance plot for the final model.
    st.image(
        "shap_feature_importance.png",
        caption="SHAP Feature Importance for the Final Model",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
