import pandas as pd
import plotly.graph_objects as go
import requests
import plotly.express as px
import numpy as np

def get_newly_enrolled(dfs):
    # Relevant columns
    newly_enrolled_1920 = dfs[0].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2021 = dfs[1].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2122 = dfs[2].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2223 = dfs[3].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]

    # concatenate all enrollment data
    df_enrolled = pd.concat([newly_enrolled_1920, newly_enrolled_2021,
                             newly_enrolled_2122, newly_enrolled_2223])

    # define cycle labels
    ciclos = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]

    df_enrolled.rename(columns={
        "check-TOTALES-H-NI": "Men",
        "check-TOTALES-M-NI": "Women"
    }, inplace=True)

    df_enrolled.index = ciclos
    df_enrolled["Cycle"] = ciclos

    # long format
    df_melted = df_enrolled.melt(
        id_vars=["Cycle"],
        var_name="Gender Identity",
        value_name="Number of enrolled"
    )

    custom_colors = ["#d8b365", "#5ab4ac"]

    # bar chart
    fig = px.bar(
        df_melted,
        x="Cycle",
        y="Number of enrolled",
        color="Gender Identity",
        color_discrete_sequence=custom_colors,
        title="New enrollment per cycle - HighSchool ITESO",
        barmode="stack"
    )

    # dotted line after the first cycle (2020-2021) (COVID -19) PAndemic
    fig.add_shape(
        type="line",
        x0="2020-2021",
        x1="2020-2021",
        y0=0,
        y1=400,
        line=dict(color="black", width=2, dash="dot")
    )

    # pandemic annotation
    fig.add_annotation(
        x="2020-2021",
        y=300,
        text="HighSchool ITESO starts\nafter COVID-19 Pandemic",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="white"
    )

    # layout
    fig.update_layout(
        title={
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=15, family="Arial")
        },
        xaxis=dict(
            title="Cycle",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        yaxis=dict(
            title="Number of Students",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        legend=dict(
            title="Gender Identity",
            title_font=dict(size=10, family="Arial"),
            font=dict(size=10, family="Arial")
        ),
        font=dict(family="Arial"),
        width=800,
        height=500,
        template="ggplot2"
    )

    fig.show()


def get_cycle_cost_graph(df_1920, df_2021):
    dfs = {
        "2019-2020": df_1920,
        "2020-2021": df_2021
    }

    # relevant cost columns
    cost_columns = [
        # 'Gasto-utiles',
        # 'Gasto-uniformes',
        # 'Gasto-cuotasvoluntarias',
        'Gasto-inscripcion',
        'Gasto-colegiatura'
        # 'Gasto-transporte'
    ]

    # custom color map
    color_map = {
        'Gasto-utiles': '#1f77b4',
        'Gasto-uniformes': '#ff7f0e',
        'Gasto-cuotasvoluntarias': '#2ca02c',
        'Gasto-inscripcion': '#dfc27d',
        'Gasto-colegiatura': '#80cdc1',
        'Gasto-transporte': '#8c564b'
    }

    processed_dfs = []

    for cycle, df in dfs.items():
        df_copy = df.copy()

        # forcing errors to NaN
        for col in cost_columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        # calculate the average cost for each cycle
        cost_per_cycle = df_copy[cost_columns].mean().reset_index()
        cost_per_cycle.columns = ["Cost Type", "Mean Cost"]
        cost_per_cycle["Cycle"] = cycle

        processed_dfs.append(cost_per_cycle)

    # concatenate all cycle DataFrames
    final_df = pd.concat(processed_dfs, ignore_index=True)

    # bar chart
    fig = px.bar(
        final_df,
        x="Cycle",
        y="Mean Cost",
        color="Cost Type",
        title="Mean Cost per Cycle - ITESO High School",
        barmode="stack",
        color_discrete_map=color_map
    )

    # layout
    fig.update_layout(
        title={
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=15, family="Arial")
        },
        xaxis=dict(
            title="Cycle",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        yaxis=dict(
            title="Mean Cost (MXN)",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        font=dict(family="Arial"),
        width=900,
        height=500,
        template="ggplot2"
    )

    fig.show()

    
def get_profile_df(df):
    # Relevant Columns
    profile = df.loc[:,['Institucion',
                                'Escuela',
                                'Estado',
                                'Municipio',
                                'Localidad', #KeyError: "['Direccion', 'Nivel', 'Subnivel', 'Carrera'] not in index"
                                'Control',
                                'Duracion_anios',
                                'Modalidad']]
    return profile

def get_newly_enrolled_map(df, cycle):
    # Get Mexico's GeoJSON 
    geojson_url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
    response = requests.get(geojson_url)
    geojson_mexico = response.json() if response.status_code == 200 else None
    
    # Relevant Columns
    columnas_estados = [
        "matricula-lugarresidencia-hom-pais-aguascalientes",
        "matricula-lugarresidencia-muj-pais-aguascalientes",
        "matricula-lugarresidencia-hom-pais-bajacalifornia",
        "matricula-lugarresidencia-muj-pais-bajacalifornia",
        "matricula-lugarresidencia-hom-pais-bajacaliforniasur",
        "matricula-lugarresidencia-muj-pais-bajacaliforniasur",
        "matricula-lugarresidencia-hom-pais-campeche",
        "matricula-lugarresidencia-muj-pais-campeche",
        "matricula-lugarresidencia-hom-pais-coahuila",
        "matricula-lugarresidencia-muj-pais-coahuila",
        "matricula-lugarresidencia-hom-pais-colima",
        "matricula-lugarresidencia-muj-pais-colima",
        "matricula-lugarresidencia-hom-pais-chiapas",
        "matricula-lugarresidencia-muj-pais-chiapas",
        "matricula-lugarresidencia-hom-pais-chihuahua",
        "matricula-lugarresidencia-muj-pais-chihuahua",
        "matricula-lugarresidencia-hom-pais-cdmx",
        "matricula-lugarresidencia-muj-pais-cdmx",
        "matricula-lugarresidencia-hom-pais-durango",
        "matricula-lugarresidencia-muj-pais-durango",
        "matricula-lugarresidencia-hom-pais-guanajuato",
        "matricula-lugarresidencia-muj-pais-guanajauto",
        "matricula-lugarresidencia-hom-pais-guerrero",
        "matricula-lugarresidencia-muj-pais-guerrero",
        "matricula-lugarresidencia-hom-pais-hidalgo",
        "matricula-lugarresidencia-muj-pais-hidalgo",
        "matricula-lugarresidencia-hom-pais-jalisco",
        "matricula-lugarresidencia-muj-pais-jalisco",
        "matricula-lugarresidencia-hom-pais-mexico",
        "matricula-lugarresidencia-muj-pais-mexico",
        "matricula-lugarresidencia-hom-pais-michoacan",
        "matricula-lugarresidencia-muj-pais-michoacan",
        "matricula-lugarresidencia-hom-pais-morelos",
        "matricula-lugarresidencia-muj-pais-morelos",
        "matricula-lugarresidencia-hom-pais-nayarit",
        "matricula-lugarresidencia-muj-pais-nayarit",
        "matricula-lugarresidencia-hom-pais-nuevoleon",
        "matricula-lugarresidencia-muj-pais-nuevoleon",
        "matricula-lugarresidencia-hom-pais-oaxaca",
        "matricula-lugarresidencia-muj-pais-oaxaca",
        "matricula-lugarresidencia-hom-pais-puebla",
        "matricula-lugarresidencia-muj-pais-puebla",
        "matricula-lugarresidencia-hom-pais-queretaro",
        "matricula-lugarresidencia-muj-pais-queretaro",
        "matricula-lugarresidencia-hom-pais-quintanaroo",
        "matricula-lugarresidencia-muj-pais-quintanaroo",
        "matricula-lugarresidencia-hom-pais-sanluispotosi",
        "matricula-lugarresidencia-muj-pais-sanluispotosi",
        "matricula-lugarresidencia-hom-pais-sinaloa",
        "matricula-lugarresidencia-muj-pais-sinaloa",
        "matricula-lugarresidencia-hom-pais-sonora",
        "matricula-lugarresidencia-muj-pais-sonora",
        "matricula-lugarresidencia-hom-pais-tabasco",
        "matricula-lugarresidencia-muj-pais-tabasco",
        "matricula-lugarresidencia-hom-pais-tamaulipas",
        "matricula-lugarresidencia-muj-pais-tamaulipas",
        "matricula-lugarresidencia-hom-pais-tlaxcala",
        "matricula-lugarresidencia-muj-pais-tlaxcala",
        "matricula-lugarresidencia-hom-pais-veracruz",
        "matricula-lugarresidencia-muj-pais-veracruz",
        "matricula-lugarresidencia-hom-pais-yucatan",
        "matricula-lugarresidencia-muj-pais-yucatan",
        "matricula-lugarresidencia-hom-pais-zacatecas",
        "matricula-lugarresidencia-muj-pais-zacatecas"
    ]
    
    # Process strings --------------------------------------------------------------------------------
    estados = [col.split('-pais-')[-1].capitalize() for col in columnas_estados]
    
    # Total Alumni per state
    total_alumnos_estado = pd.DataFrame({
        "Estado": estados,
        "Total Alumnos": df[columnas_estados].sum()
    })
    
    # Correction names for some states
    estado_mapper = {
        "Cdmx": "Ciudad de México",
        "Mexico": "México",
        "Michoacan": "Michoacán",
        "CDMX": "Ciudad de México",
        "Nuevoleon": "Nuevo León",
        "Queretaro": "Querétaro",
        "Sanluispotosi": "San Luis Potosí",
        "Yucatan": "Yucatán",
        "Bajacalifornia": "Baja California",
        "Bajacaliforniasur": "Baja California Sur",
        "Quintanaroo":"Quintana Roo"
    }
    total_alumnos_estado["Estado"] = total_alumnos_estado["Estado"].replace(estado_mapper)
    
    # Map ---------------------------------------------------------------------------------------
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson_mexico,
        locations=total_alumnos_estado["Estado"],
        featureidkey="properties.name",
        z=total_alumnos_estado["Total Alumnos"],
        colorscale="balance", # This color scale is the one that fit the best
        colorbar_title="Total Alumnos",
        marker_opacity=0.7,
        marker_line_width=0.5
    ))
    
    fig.update_layout(
        title="Distribution of First Newly Enrolled per State (ITESO "+cycle+")",
        mapbox=dict(
            style="carto-positron",
            center={"lat": 23.6345, "lon": -102.5528},
            zoom=4.5
        ),
        margin={"r":0, "t":50, "l":0, "b":0}
    )
    
    fig.show()
    return 

def get_student_flow(dfs):

    # Aux Function
    def get_graduates_df(df_year):
        df_year = df_year.copy()
        df_year['Graduates'] = df_year['matricula egresados-hom'] + df_year['matricula egresados-muj']
        df_year['Newly enrolled'] = df_year["check-TOTALES-H-NI"] + df_year["check-TOTALES-M-NI"]
        return df_year
    
    df_1920 = get_graduates_df(dfs[0])
    df_2021 = get_graduates_df(dfs[1])
    df_2122 = get_graduates_df(dfs[2])
    df_2223 = get_graduates_df(dfs[3])

    cycles = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]
    dfs = [df_1920, df_2021, df_2122, df_2223]

    data_list = []

    # sum the relevant columns for each cycle and create a new df
    for cycle, df in zip(cycles, dfs):
        df_sum = df[['Graduates', 'Newly enrolled']].sum().to_frame().T
        df_sum['Ciclo'] = cycle
        data_list.append(df_sum)

    # concatenate all DataFrames
    df_combined = pd.concat(data_list, ignore_index=True)

    # long format 
    df_long = df_combined.melt(
        id_vars=["Ciclo"],
        value_vars=['Graduates', 'Newly enrolled'],
        var_name="Variable",
        value_name="Valor"
    )

    # bar chart
    fig = px.bar(
        df_long,
        x="Ciclo",
        y="Valor",
        color="Variable",
        barmode="group",
        title="Student flow per Scholar cycle",
        labels={"Valor": "Quantity", "Ciclo": "Scholar Cycle", "Variable": "Variable"},
        color_discrete_map={
            'Graduates': '#5ab4ac',
            'Newly enrolled': '#fc8d59'
        }
    )

    # layout
    fig.update_layout(
        title={
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=15, family="Arial")
        },
        xaxis=dict(
            title="Scholar Cycle",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        yaxis=dict(
            title="",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        font=dict(family="Arial"),
        width=1000,
        height=500,
        template="ggplot2"
    )

    fig.show()

    
def get_historical_admission(dfs):
    # Relevant columns
    newly_enrolled_1920 = dfs[0].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2021 = dfs[1].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2122 = dfs[2].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2223 = dfs[3].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]

    # concatenate data from all cycles
    df_enrolled = pd.concat([newly_enrolled_1920, newly_enrolled_2021,
                             newly_enrolled_2122, newly_enrolled_2223])
    cycles = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]

    df_enrolled.rename(
        columns={
            "check-TOTALES-H-NI": "Men",
            "check-TOTALES-M-NI": "Women"
        },
        inplace=True
    )

    # assign cycle labels to the df index
    df_enrolled.index = cycles
    df_enrolled["Cycle"] = cycles

    df_enrolled["Newly Enrolled"] = df_enrolled["Men"] + df_enrolled["Women"]

    # long format 
    df_long = df_enrolled.melt(
        id_vars=["Cycle"],
        value_vars=["Newly Enrolled"],
        var_name="Variable",
        value_name="Valor"
    )

    # bar chart 
    fig = px.bar(
        df_long,
        x="Cycle",
        y="Valor",
        color="Variable",
        barmode="group",
        opacity=0.6,  # bar opacity
        title="New Enrollment per Cycle - HighSchool ITESO",
        labels={"Valor": "Quantity", "Cycle": "Scholar Cycle", "Variable": "Variable"},
        color_discrete_map={"Newly Enrolled": "#4393c3"}
    )

    # trend line
    fig.add_trace(
        go.Scatter(
            x=df_enrolled["Cycle"],
            y=df_enrolled["Newly Enrolled"],
            mode="lines+markers",
            name="Trend Line",
            line=dict(color="#fc8d59", width=2)
        )
    )

    # layout
    fig.update_layout(
        title={
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=15, family="Arial")
        },
        xaxis=dict(
            title="Cycle",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        yaxis=dict(
            title="Number of Students",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        font=dict(family="Arial"),
        width=800,
        height=500,
        template="ggplot2",
        showlegend=True
    )

    fig.show()

    
def get_historical_admission_prediction(dfs):
    newly_enrolled_1920 = dfs[0].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2021 = dfs[1].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2122 = dfs[2].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]
    newly_enrolled_2223 = dfs[3].loc[:, ["check-TOTALES-H-NI", "check-TOTALES-M-NI"]]

    # Concatenate dfs
    df_enrolled = pd.concat([newly_enrolled_1920, newly_enrolled_2021, newly_enrolled_2122, newly_enrolled_2223])
    cycles = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]

    df_enrolled.rename(columns={"check-TOTALES-H-NI": "Men", "check-TOTALES-M-NI": "Women"}, inplace=True)
    
    df_enrolled.index = cycles
    df_enrolled['Cycle'] = cycles
    
    # total admissions per 
    df_enrolled['Newly Enrolled'] = df_enrolled['Men'] + df_enrolled['Women']
    
    # (SIMPLE) linear regression (excluding the first two cycles)
    x = np.array([2, 3]) 
    y = df_enrolled.loc[["2021-2022", "2022-2023"], "Newly Enrolled"].values  # Extract last two cycles

    # fit a line (1st-degree polynomial) (Just one variable)
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)

    # Predict next cycle (2023-2024 -> x=4)
    x_next = 4
    next_value = poly(x_next)

    # Create new dataframe for prediction
    df_next_cycle = pd.DataFrame({
        "Cycle": ["2023-2024"], 
        "Newly Enrolled": [next_value]
    })

    # Concatenate all data (including predicted value)
    df_combined_pred = pd.concat([df_enrolled, df_next_cycle], ignore_index=True)

    # transform to long format
    df_long = df_combined_pred.melt(
        id_vars=["Cycle"], 
        value_vars=['Newly Enrolled'],
        var_name="Variable", 
        value_name="Valor"
    )


    color_map = {
        'Newly Enrolled': '#4393c3',  # Blue
        'Predicted': '#d6604d'  # Red 
    }

    df_long["Color"] = df_long["Cycle"].apply(
        lambda x: 'Predicted' if x == "2023-2024" else 'Newly Enrolled'
    )

    # bar chatr
    
    fig = px.bar(
        df_long, 
        x="Cycle", 
        y="Valor", 
        color="Color", 
        barmode="overlay",
        opacity=0.8,
        title="New Enrollment per Cycle - HighSchool ITESO (with Prediction)",
        labels={"Valor": "Quantity", "Cycle": "Scholar Cycle", "Color": "Variable"},
        color_discrete_map=color_map
    )

    # Center bars
    fig.update_traces(marker=dict(line=dict(width=0)), width=0.4)
    fig.update_layout(bargap=0.2, bargroupgap=0.05)


    # original trend line
    fig.add_scatter(
        x=df_combined_pred["Cycle"],
        y=df_combined_pred["Newly Enrolled"],
        mode='lines+markers',
        name='Trend Line',
        line=dict(color='#fc8d59', width=2)
    )

    # regression line
    x_pred = np.array([2, 3, 4])  # Last two cycles + prediction
    y_pred = poly(x_pred)

    fig.add_scatter(
        x=["2021-2022", "2022-2023", "2023-2024"],  
        y=y_pred,
        mode='lines+markers',
        name='Regression Line (Last Two Years)',
        line=dict(color='gray', width=2, dash='dot'),
        opacity=0.4
    )

    # layout
    fig.update_layout(
        title={
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=15, family="Arial")
        },
        xaxis=dict(
            title="Cycle",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial"),
            tickmode="array",
            tickvals=np.arange(len(df_combined_pred)),
            ticktext=df_combined_pred["Cycle"]
        ),
        yaxis=dict(
            title="Number of Students",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        font=dict(family="Arial"),
        width=800,
        height=500,
        template="ggplot2",
        showlegend=True
    )

    fig.show()
