import pandas as pd
import plotly.graph_objects as go
import requests
import plotly.express as px
import numpy as np

def newly_enrolled_graph(df, cycle):
    # group by Faculty and sum the relevant columns
    ingreso_aggregated = df.groupby("Escuela")[["primer ingreso-hom", "primer ingreso-muj"]].sum().reset_index()

    ingreso_aggregated_copy = ingreso_aggregated.copy()

    # the department names into English
    translation_dict = {
        'DEPARTAMENTO DE ECONOMIA ADMINISTRACION Y MERCADOLOGIA': 'DEPARTMENT OF ECONOMICS, ADMINISTRATION, AND MARKETING',
        'DEPTO DE HABITAT Y DESARROLLO URBANO': 'DEPARTMENT OF HABITAT AND URBAN DEVELOPMENT',
        'DEPARTAMENTO DE PROCESOS TECNOLOGICOS E INDUSTRIALES': 'DEPARTMENT OF TECHNOLOGICAL AND INDUSTRIAL PROCESSES',
        'DEPTO DE ESTUDIOS SOCIO CULTURALES': 'DEPARTMENT OF SOCIO-CULTURAL STUDIES',
        'DEPARTAMENTO DE ESTUDIOS SOCIOPOLITICOS Y JURIDICOS': 'DEPARTMENT OF SOCIO-POLITICAL AND LEGAL STUDIES',
        'DEPTO DE ELECTRONICA SISTEMAS E INFORMATICA': 'DEPARTMENT OF ELECTRONICS, SYSTEMS, AND INFORMATICS',
        'DEPARTAMENTO DE PSICOLOGIA EDUCACION Y SALUD': 'DEPARTMENT OF PSYCHOLOGY, EDUCATION, AND HEALTH',
        'DEPARTAMENTO DE MATEMATICAS Y FISICA': 'DEPARTMENT OF MATHEMATICS AND PHYSICS',
        'DEPTO DE FILOSOFIA Y HUMANIDADES': 'DEPARTMENT OF PHILOSOPHY AND HUMANITIES'
    }
    
    # apply the translation 
    ingreso_aggregated_copy["Escuela"] = ingreso_aggregated_copy["Escuela"].replace(translation_dict)
    
    ingreso_aggregated_copy.rename(
        columns={
            "Escuela": "Faculty",
            "primer ingreso-hom": "Men",
            "primer ingreso-muj": "Women"
        },
        inplace=True
    )
    
    # calculate the total of enrolled students per faculty 
    ingreso_aggregated_copy["total"] = ingreso_aggregated_copy["Men"] + ingreso_aggregated_copy["Women"]
    ingreso_aggregated_copy = ingreso_aggregated_copy.sort_values(by="total", ascending=False)

    # Long Format
    ingreso_melted = ingreso_aggregated_copy.melt(
        id_vars=["Faculty", "total"],
        var_name="Gender Identity",
        value_name="Number of enrolled"
    )

    custom_colors = ["#d8b365", "#5ab4ac"]

    # bar chart
    fig = px.bar(
        ingreso_melted, 
        x="Number of enrolled", 
        y="Faculty", 
        color="Gender Identity", 
        orientation="h",
        color_discrete_sequence=custom_colors,
        title="New enrollment per faculty - ITESO 2019-2020",
        category_orders={"Faculty": ingreso_aggregated_copy["Faculty"].tolist()}
    )

    # aArial font 
    fig.update_layout(
        title={
            "text": "New enrollment per faculty - ITESO " + cycle,
            "y": 0.83,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=15, family="Arial")
        },
        xaxis=dict(
            title="Number of Students",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        yaxis=dict(
            title="",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        legend=dict(
            title="Gender Identity",
            title_font=dict(size=10, family="Arial"),
            font=dict(size=10, family="Arial")
        ),
        font=dict(family="Arial"),
        width=1000,
        height=400,
        template="ggplot2"
    )

    fig.show()

    return


def get_newly_enrolled_map(df, cycle):
    # Get Mexico's GeoJSON 
    geojson_url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
    response = requests.get(geojson_url)
    geojson_mexico = response.json() if response.status_code == 200 else None

    cycle_dict = {"2019-2020":"ciclo 19 a 20",
                  "2020-2021":"ciclo 19 a 20",
                  "2021-2022":"ciclo 19 a 20",
                  "2022-2023":"ciclo 19 a 20",}
    # Relevant Columns
    columnas_estados = [
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Aguascalientes',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Baja California',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Baja California Sur',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Campeche',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Coahuila',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Colima',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Chiapas',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Chihuahua',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- CDMX',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Durango',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Guanajuato',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Guerrero',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Hidalgo',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Jalisco',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Mexico ',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Michoacan',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Morelos',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Nayarit',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Nuevo Leon',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Oaxaca',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Puebla',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Queretaro',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Quintana Roo',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- San Luis Potosi ',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Sinaloa',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Sonora',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Tabasco',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Tamaulipas',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Tlaxcala',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Veracruz',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Yucatan',
     '# de alumnos primer ingreso carrera -'+cycle_dict[cycle]+' - lug bachill- Zacatecas'
    ]

    # Process strings --------------------------------------------------------------------------------
    
    columnas_estados_limpias = [col.split("- lug bachill- ")[-1].strip() for col in columnas_estados]

    # Total Alumni per state
    total_alumnos_estado = pd.DataFrame({
        "Estado": columnas_estados_limpias,
        "Total Alumnos": df[columnas_estados].sum()
    })

    # Correction names for some states
    estado_mapper = {
        "Mexico": "México",
        "Michoacan": "Michoacán",
        "CDMX": "Ciudad de México",
        "Nuevo Leon": "Nuevo León",
        "Queretaro": "Querétaro",
        "San Luis Potosi": "San Luis Potosí",
        "Yucatan": "Yucatán"
    }
    total_alumnos_estado["Estado"] = total_alumnos_estado["Estado"].replace(estado_mapper)

    color_scale = "balance" # This color scale is the one that fit the best

    # Map ---------------------------------------------------------------------------------------
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson_mexico,
        locations=total_alumnos_estado["Estado"],
        featureidkey="properties.name",  # Cambio de NOM_ENT a name
        z=total_alumnos_estado["Total Alumnos"],
        colorscale=color_scale,
        colorbar_title="Total Alumnos",
        marker_opacity=0.7,
        marker_line_width=0.5
    ))

    fig.update_layout(
        title="Distribution of First Newly Enrolled per State (ITESO "+cycle+")",
        mapbox=dict(
            style="carto-positron",  # Basemap
            center={"lat": 23.6345, "lon": -102.5528},  # Center in Mexico
            zoom=4.5
        ),
        margin={"r":0, "t":50, "l":0, "b":0}
    )

    fig.show()  
    
def get_faculty_cost_graph(df, cycle):
 
    under_grad_df = df.copy()  # Avoid modifying original DF

    # Convert columns to numeric
    cols_to_sum = [
        '$ gasto promedio anual - cuotas voluntarias -ciclo 19 a 20',
        '$ gasto promedio anual - inscripcion -ciclo 19 a 20',
        '$ gasto promedio anual- colegiatura -ciclo 19 a 20',
        '$ gasto promedio anual - materiales educativos e insumos -ciclo 19 a 20'
    ]

    # Convert to numeric (force errors to NaN)
    for col in cols_to_sum:
        under_grad_df[col] = pd.to_numeric(under_grad_df[col], errors='coerce')

    under_grad_df['mean cost'] = under_grad_df[cols_to_sum].sum(axis=1)

    # Group by school and calculate the average total annual cost
    avg_cost_per_school = under_grad_df.groupby('Escuela')['mean cost'].mean().reset_index()

    cost_aggregated_copy = avg_cost_per_school.copy()

    # Translation dictionary for faculties
    translation_dict = {
        "DEPARTAMENTO DE ECONOMIA ADMINISTRACION Y MERCADOLOGIA": "DEPARTMENT OF ECONOMICS, ADMINISTRATION, AND MARKETING",
        "DEPTO DE HABITAT Y DESARROLLO URBANO": "DEPARTMENT OF HABITAT AND URBAN DEVELOPMENT",
        "DEPARTAMENTO DE PROCESOS TECNOLOGICOS E INDUSTRIALES": "DEPARTMENT OF TECHNOLOGICAL AND INDUSTRIAL PROCESSES",
        "DEPTO DE ESTUDIOS SOCIO CULTURALES": "DEPARTMENT OF SOCIO-CULTURAL STUDIES",
        "DEPARTAMENTO DE ESTUDIOS SOCIOPOLITICOS Y JURIDICOS": "DEPARTMENT OF SOCIO-POLITICAL AND LEGAL STUDIES",
        "DEPTO DE ELECTRONICA SISTEMAS E INFORMATICA": "DEPARTMENT OF ELECTRONICS, SYSTEMS, AND INFORMATICS",
        "DEPARTAMENTO DE PSICOLOGIA EDUCACION Y SALUD": "DEPARTMENT OF PSYCHOLOGY, EDUCATION, AND HEALTH",
        "DEPARTAMENTO DE MATEMATICAS Y FISICA": "DEPARTMENT OF MATHEMATICS AND PHYSICS",
        "DEPTO DE FILOSOFIA Y HUMANIDADES": "DEPARTMENT OF PHILOSOPHY AND HUMANITIES"
    }

    # Apply translation 
    cost_aggregated_copy["Escuela"] = cost_aggregated_copy["Escuela"].replace(translation_dict)

    cost_aggregated_copy.rename(columns={
        "Escuela": "Faculty",
        "mean cost": "Mean Cost"
    }, inplace=True)

    # Sort by Mean Cost (Descending)
    cost_aggregated_copy = cost_aggregated_copy.sort_values(by="Mean Cost", ascending=False)

    mean_cost = cost_aggregated_copy["Mean Cost"].mean() # Compute the mean cost line

    custom_colors = ["#5ab4ac"]

    # Bar chart ---------------------------------------------------------------------------------------
    
    fig = px.bar(
        cost_aggregated_copy, 
        x="Mean Cost", 
        y="Faculty", 
        orientation="h",
        color_discrete_sequence=custom_colors,
        title=f"Mean Cost per Faculty - ITESO {cycle}",
        category_orders={"Faculty": cost_aggregated_copy["Faculty"].tolist()}  # Order bars by cost
    )
    
    # Add mean cost line
    fig.add_shape(
        type="line",
        x0=mean_cost, x1=mean_cost, 
        y0=-0.5, y1=len(cost_aggregated_copy["Faculty"]) - 0.5,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add annotation for mean cost
    fig.add_annotation(
        x=mean_cost + 25000,
        y=len(cost_aggregated_copy["Faculty"]) - 5,  
        text=f"Mean: ${mean_cost:,.0f} MXN",
        showarrow=False,
        font=dict(size=9, color="black")
    )

    # Apply layout settings
    fig.update_layout(
        title={
            "text": f"Mean Cost per Faculty - ITESO {cycle}",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=15, family="Arial")
        },
        xaxis=dict(
            title="Mean Cost (MXN)",
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

    # Set axis range
    fig.update_xaxes(range=[0, cost_aggregated_copy["Mean Cost"].max() * 1.1])

    fig.show()
    
def get_profile_df(df):
   # Select relevant columns
    profile = df.loc[:,['Institucion',
                                'Escuela',
                                'Estado',
                                'Municipio',
                                'Localidad',
                                'Direccion',
                                'Control',
                                'Nivel',
                                'Subnivel',
                                'Carrera',
                                'Modalidad']]
    return profile

def student_flow_graph(dfs):
    # Create copies of the provided dfs
    df_1920 = dfs[0].copy()
    df_2021 = dfs[1].copy()
    df_2122 = dfs[2].copy()
    df_2223 = dfs[3].copy()

    # List of academic cycles and their dfs
    cycles = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]
    dfs = [df_1920, df_2021, df_2122, df_2223]

    data_list = []

    # Sum the relevant columns for each cycle and organize them into a new df
    for cycle, df in zip(cycles, dfs):
        df_sum = df[['# de alumnos egresados-tot', 'V90', 'lugares ofertados']].sum().to_frame().T
        df_sum.rename(columns={
            'V90': 'Newly enrolled',
            'lugares ofertados': 'Spots Offered',
            '# de alumnos egresados-tot': 'Graduates'
        }, inplace=True)
        df_sum['Ciclo'] = cycle
        data_list.append(df_sum)

    # Concatenate all the dfs
    df_combined = pd.concat(data_list, ignore_index=True)

    # Transform the DataFrame into a long format
    df_long = df_combined.melt(
        id_vars=["Ciclo"],
        value_vars=['Graduates', 'Newly enrolled', 'Spots Offered'],
        var_name="Variable",
        value_name="Valor"
    )

    # bar chart ---------------------------------------------------------------------------------------
    fig = px.bar(
        df_long,
        x="Ciclo",
        y="Valor",
        color="Variable",
        barmode="group",
        title="Student flow per Scholar cycle Grad School",
        labels={"Valor": "Quantity", "Ciclo": "Scholar Cycle", "Variable": "Variable"},
        color_discrete_map={
            'Graduates': '#5ab4ac',
            'Spots Offered': '#d8b365',
            'Newly enrolled': '#fc8d59'
        }
    )

    # Customize layout
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
            title="Quantity",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        font=dict(family="Arial"),
        width=1000,
        height=500,
        template="ggplot2"
    )

    fig.show()

    
def get_historical_grow(dfs):
    
    df_1920 = dfs[0].copy()
    df_2021 = dfs[1].copy()
    df_2122 = dfs[2].copy()
    df_2223 = dfs[3].copy()

    # list of cycles and their corresponding dfs
    cycles = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]
    dfs = [df_1920, df_2021, df_2122, df_2223]

    data_list = []

    # sum the relevant columns for each cycle and organize them 
    for cycle, df in zip(cycles, dfs):
        df_sum = df[['V90']].sum().to_frame().T
        df_sum.rename(columns={'V90': 'Newly enrolled'}, inplace=True)
        df_sum['Ciclo'] = cycle
        data_list.append(df_sum)

    df_combined = pd.concat(data_list, ignore_index=True)

    # transform to a long format 
    df_long = df_combined.melt(
        id_vars=["Ciclo"], 
        value_vars=['Newly enrolled'],
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
        opacity=0.6,  # bar opacity
        title="Admissions Hitorical Growth",
        labels={"Valor": "Cantidad", "Ciclo": "Scholar Cycle", "Variable": "Variable"},
        color_discrete_map={
            'Graduates': '#5ab4ac',
            'Spots Offered': '#d8b365',
            'Newly enrolled': '#4393c3'
        }
    )

    # add a line on top of the bars
    fig.add_scatter(
        x=df_combined["Ciclo"], 
        y=df_combined["Newly enrolled"], 
        mode='lines+markers', 
        name='Trend Line', 
        line=dict(color='#fc8d59', width=2)
    )

    # add a dotted line after the first cycle (2020-2021) (COVID -19 PAndemic)
    fig.add_shape(
        type="line",
        x0="2020-2021", x1="2020-2021",
        y0=0, y1=3100,  
        line=dict(color="black", width=2, dash="dot")
    )

    # add a "pandemic" annotation 
    fig.add_annotation(
        x="2020-2021",
        y=2500,
        text="Covid-19 Pandemic",
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
            title="Scholar Cycle",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial")
        ),
        yaxis=dict(
            title="Quantity",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial"),
            range=[0, 3100]
        ),
        font=dict(family="Arial"),
        width=1000,
        height=500,
        template="ggplot2"
    )

    fig.show()


def get_historical_grow_prediction(dfs):

    df_1920 = dfs[0].copy()
    df_2021 = dfs[1].copy()
    df_2122 = dfs[2].copy()
    df_2223 = dfs[3].copy()

    # list of cycles and their respective dfs
    cycles = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]
    original_dfs = [df_1920, df_2021, df_2122, df_2223]

    # sum the values in each dataframe and store them
    data_list = []
    for cycle, df in zip(cycles, original_dfs):
        df_sum = df[['V90']].sum().to_frame().T
        df_sum.rename(columns={'V90': 'Newly enrolled'}, inplace=True)
        df_sum['Ciclo'] = cycle
        data_list.append(df_sum)

    # concatenate everything into a single dataframe
    df_combined = pd.concat(data_list, ignore_index=True)

    # (SIMPLE) linear regression (excluding the first cycle)
    x = np.arange(1, len(df_combined))  # [1, 2, 3] (excluding 2019-2020)
    y = df_combined.loc[1:, "Newly enrolled"].values  # exclude the first cycle

    # fit a line (1st-degree polynomial) (Just one variable)
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)

    # predict the next cycle (2023-2024)
    x_next = 4
    next_value = poly(x_next)

    # create a new dataframe with the next cycle's prediction
    df_next_cycle = pd.DataFrame({
        "Ciclo": ["2023-2024"], 
        "Newly enrolled": [next_value]
    })

    # concatenate to have all data (including the prediction)
    df_combined_pred = pd.concat([df_combined, df_next_cycle], ignore_index=True)

    # transform to long format
    df_long = df_combined_pred.melt(
        id_vars=["Ciclo"], 
        value_vars=['Newly enrolled'],
        var_name="Variable", 
        value_name="Valor"
    )

    # center bars and differentiate regression
    color_map = {
        'Newly enrolled': '#4393c3',
        'Regression Prediction': '#d6604d'
    }

    df_long["Color"] = df_long["Ciclo"].apply(
        lambda x: 'Regression Prediction' if x == "2023-2024" else 'Newly enrolled'
    )

    fig = px.bar(
        df_long, 
        x="Ciclo", 
        y="Valor", 
        color="Color", 
        barmode="overlay",  # overlay instead of grouping
        opacity=0.8, 
        title="Admissions Historical Growth (with Trend and Regression)",
        labels={"Valor": "Cantidad", "Ciclo": "Scholar Cycle", "Color": "Variable"},
        color_discrete_map=color_map
    )

    # adjust bar spacing to center them
    fig.update_traces(marker=dict(line=dict(width=0)), width=0.4)
    fig.update_layout(bargap=0.2, bargroupgap=0.05)

    # original trend line
    fig.add_scatter(
        x=df_combined_pred["Ciclo"],
        y=df_combined_pred["Newly enrolled"],
        mode='lines+markers',
        name='Trend Line',
        line=dict(color='#fc8d59', width=2)
    )

    # regression line
    x_pred = np.arange(1, len(df_combined) + 1)  # [1, 2, 3, 4]
    y_pred = poly(x_pred)

    fig.add_scatter(
        x=df_combined_pred["Ciclo"][1:],  # from the second cycle onward
        y=y_pred,
        mode='lines+markers',
        name='Regression Line',
        line=dict(color='gray', width=2, dash='dot'),
        opacity=0.3
    )

    # dotted line in 2020-2021 (pandemic)
    fig.add_shape(
        type="line",
        x0="2020-2021", x1="2020-2021",
        y0=0, y1=2200,  
        line=dict(color="black", width=2, dash="dot")
    )

    fig.add_annotation(
        x="2020-2021",
        y=1800,
        text="Covid-19 Pandemic",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="white"
    )

    # Layout
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
            tickfont=dict(size=10, family="Arial"),
            tickmode="array",
            tickvals=np.arange(len(df_combined_pred)),
            ticktext=df_combined_pred["Ciclo"]
        ),
        yaxis=dict(
            title="Quantity",
            title_font=dict(size=10, family="Arial"),
            tickfont=dict(size=10, family="Arial"),
            range=[0, max(df_combined_pred["Newly enrolled"])*1.1]
        ),
        font=dict(family="Arial"),
        width=1000,
        height=500,
        template="ggplot2"
    )

    fig.show()

   