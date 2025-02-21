import pandas as pd
import numpy as np
import plotly.express as px

def get_cost_competitors(df):
    # Relevant columns
    cost_cols = [
        #'$ gasto promedio anual - cuotas voluntarias -ciclo 19 a 20',
        #'$ gasto promedio anual - inscripcion -ciclo 19 a 20',
        '$ gasto promedio anual- colegiatura -ciclo 19 a 20'#,
        #'$ gasto promedio anual - materiales educativos e insumos -ciclo 19 a 20'
    ]

    # Group by institutions
    df_colegiatura = df.groupby('Institucion', as_index=False)[cost_cols].mean()

    df_colegiatura = df_colegiatura.sort_values('$ gasto promedio anual- colegiatura -ciclo 19 a 20', ascending=False)

    return df_colegiatura.head(10) # Just return the top 10 

def get_growth_perc(dfs):

    # group by 'Institucion' 
    df_1 = dfs[0].groupby('Institucion', as_index=False)['V90'].sum()
    df_1.rename(columns={'V90': 'V90_1'}, inplace=True)

    df_2 = dfs[1].groupby('Institucion', as_index=False)['V90'].sum()
    df_2.rename(columns={'V90': 'V90_2'}, inplace=True)

    df_3 = dfs[2].groupby('Institucion', as_index=False)['V90'].sum()
    df_3.rename(columns={'V90': 'V90_3'}, inplace=True)

    # merge the three dfs on 'Institucion'
    df_agg = pd.merge(df_1, df_2, on='Institucion', how='outer')
    df_agg = pd.merge(df_agg, df_3, on='Institucion', how='outer')

    # 3) fill missing values with 0 
    df_agg[['V90_1', 'V90_2', 'V90_3']] = df_agg[['V90_1', 'V90_2', 'V90_3']].fillna(0)

    # calculate growth ratios, avoiding division by zero
    df_agg['mean1'] = np.where(df_agg['V90_1'] == 0, 0, df_agg['V90_2'] / df_agg['V90_1'])
    df_agg['mean2'] = np.where(df_agg['V90_2'] == 0, 0, df_agg['V90_3'] / df_agg['V90_2'])

    # compute the average of the two ratios
    df_agg['growth mean'] = (df_agg['mean1'] + df_agg['mean2']) / 2

    # return columns of interest
    return df_agg[['Institucion', 'growth mean']]#.sort_values('growth mean', ascending=False)

def get_growth_graph(df_tuition, dfs):
    # Calculate growth percentages for all institutions
    df_growth = get_growth_perc(dfs)
    
    # top 10 by cost + University of Guadalajara
    institution_list = df_tuition.head(10).Institucion.to_list() + ['UNIVERSIDAD DE GUADALAJARA']
    
    df_filtered = df_growth[df_growth["Institucion"].isin(institution_list)].copy()
    
    df_filtered.rename(columns={'Institucion': 'Institution', 'growth mean': 'Growth Mean'}, inplace=True)

    df_filtered.sort_values(by='Growth Mean', ascending=False, inplace=True)

    # Calculate the overall mean growth 
    mean_growth = df_filtered['Growth Mean'].mean()

    # Mark ITESO vs. Competitors
    df_filtered['Institutions'] = df_filtered['Institution'].apply(
        lambda x: 'ITESO' if x.upper() == 'ITESO' else 'Competitors'
    )

    color_map = {
        'ITESO': '#d8b365',       
        'Competitors': '#5ab4ac'  
    }

    # bar chart
    fig = px.bar(
        df_filtered,
        x="Growth Mean",
        y="Institution",
        orientation="h",
        color='Institutions',
        color_discrete_map=color_map,
        title="Mean Enrollment Growth per Institution",
        category_orders={"Institution": df_filtered["Institution"].tolist()}  # preserve sorting order
    )

    # average growth line
    fig.add_shape(
        type="line",
        x0=mean_growth,
        x1=mean_growth,
        y0=-0.5,
        y1=len(df_filtered) - 0.5,
        line=dict(color="red", width=2, dash="dash")
    )

    # annotation for the mean value
    fig.add_annotation(
        x=mean_growth,
        y=len(df_filtered) - 6,  # placement depends on the number of rows
        text=f"Mean: {mean_growth:.2f}",
        showarrow=False,
        font=dict(size=9, color="black"),
        xshift=60  # shift label to avoid overlap
    )

    # layout 
    fig.update_layout(
        title={
            "text": "Mean Enrollment Growth per Institution",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=15, family="Arial")
        },
        xaxis=dict(
            title="Mean Enrollment",
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

    # x-axis range 
    max_growth = df_filtered["Growth Mean"].max()
    fig.update_xaxes(range=[0, max_growth * 1.1])

    fig.show()

    
def treemap_institutions(jal_year, cycle):
    # group by institution 
    df_agg = jal_year.groupby('Institucion', as_index=False)[['V90']].sum()

    df_agg = df_agg.sort_values('V90', ascending=False)

    df_agg.reset_index(inplace=True)
    df_agg['color_group'] = df_agg['Institucion'].apply(lambda x: 'ITESO' if x == 'ITESO' else 'OTRAS')

    color_map = {
        'ITESO': '#d6604d',
        'OTRAS': '#4393c3'
    }

    # treemap 
    fig = px.treemap(
        df_agg,
        path=['Institucion'],
        values='V90',
        color='color_group',
        color_discrete_map=color_map,
        title="Distribution of Newly Enrolled Students Across Jalisco Schools " + cycle
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
            tickfont=dict(size=10, family="Arial")
        ),
        font=dict(family="Arial"),
        width=1000,
        height=500,
        template="ggplot2"
    )


    fig.update_traces(
        textinfo='label+value+percent root',
        hovertemplate=(
            '<b>%{label}</b><br>Students: %{value}<br>'
            'Percentage of Total: %{percentRoot:.2%}<extra></extra>' 
       ), #hover text
        textfont=dict(color='white')
    )

    fig.show()
