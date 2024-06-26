import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import requests
import json
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from bokeh.models import HoverTool, Legend, ColumnDataSource, CustomJS, CDSView, GroupFilter, BooleanFilter, Circle, Span
from bokeh.palettes import turbo
from bokeh.plotting import figure, output_file, show, save
from config import Settings

app_settings = Settings()


def component_visual(pca, column_names):
    """
    This function retrieves and visualizes the PCA loadings.
    :param pca: the principal component analysis
    :param column_names: list with column names
    :return: plot_figure = bokehplot
    """
    plot_figure = figure(
        x_axis_label="Scan nummer",
        y_axis_label="Intensiteit",
        width=1700, 
        height=900, 
        tools='pan, save, reset, box_zoom'
    )
    colors = turbo(11)
    color_count=1

    for filename in [f"{app_settings.DATA_PATH}training_whiskey_A1_1",
                     f"{app_settings.DATA_PATH}training_whiskey_B1_1",
                     f"{app_settings.DATA_PATH}training_whiskey_C1_1",
                     f"{app_settings.DATA_PATH}training_whiskey_D1_1",
                     f"{app_settings.DATA_PATH}training_whiskey_E1_1",
                     f"{app_settings.DATA_PATH}training_whiskey_A_B1_90_10_1",
                     f"{app_settings.DATA_PATH}training_whiskey_A_B1_80_20_1",
                     f"{app_settings.DATA_PATH}training_whiskey_A_B1_50_50_1",
                     f"{app_settings.DATA_PATH}training_whiskey_A_B1_20_80_1",
                     f"{app_settings.DATA_PATH}training_whiskey_A_B1_10_90_1",]:
        ms = pickle.load(open(filename, 'rb'))
        label = filename.split("/")[-1].strip("training_").strip("whiskey_")
        tic_inten = ms.intensity_array
        plot_figure.line(list(range(len(tic_inten))), tic_inten,
        line_color = colors[color_count],
        legend_label = label,
        line_width=2,
        line_alpha=0.5,
        muted_alpha=0)
        color_count+=1
    # The 50 greatest loadings for PC1 (comp0) and PC2 (comp1)
    comp0 = [[x,_] for _,x in sorted(zip(pca.components_[0].tolist(),column_names), key=lambda inner_list: abs(inner_list[0]))]
    comp1 = [[x,_] for _,x in sorted(zip(pca.components_[1].tolist(),column_names), key=lambda inner_list: abs(inner_list[0]))]
    
    # Use [ for idx in [item[0] for item in comp0][-50:]: ] for PC1 loadings and use [ for idx in [item[0] for item in comp1][-50:]: ] for PC2 loadings
    # for idx in [item[0] for item in comp0][-50:]:
    for idx in [item[0] for item in comp1][-50:]:
        vline = Span(location=idx, dimension='height', line_color='black', line_width=2, line_alpha=0.5)
        plot_figure.renderers.extend([vline])

    plot_figure.legend.location = "top_right"
    plot_figure.legend.click_policy="mute"
    output_file("/exports/nas/wilbrink.a/whisky_project/whisky_plotjes/PC2_700-8000_AvsB.html")
    return plot_figure


def pca_visual(x, df, expl_variance):
    """
    This function visualizes the PCA in a scatter plot with bokeh.
    :param x: data on which dimensionality reduction has been performed
    :param df: dataframe with data before dimensionality reduction
    :param expl_variance: a statistical measure of how much variation 
     in the dataset can be attributed to each of the principal components
    :return: plot_figure = bokehplot of the PCA
    """
    final_df = pd.DataFrame(x, columns=('x', 'y'))
    final_df['ids'] = df.index
    final_df['label'] = df['code']
    final_df['group'] = df['group']

    colors = turbo(len(final_df["group"].unique()))
    plot_figure = figure(
        x_axis_label="PC1 (" + str(round(expl_variance[0]*100, 2)) + "%)",
        y_axis_label="PC2 (" + str(round(expl_variance[1]*100, 2)) + "%)",
        width= 1500, #1150,
        height= 850, #625,
        sizing_mode='scale_width',
        tools='pan, save, reset, box_zoom'
    )
    plot_figure.min_border_top = 10
    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <span style='font-size: 16px; color: #224499'>Whisky:</span>
            <span style='font-size: 16px'>@label</span>
        </div>
    </div>
    """))

    cds = ColumnDataSource(final_df)
    color_count = 0
    for x in final_df["group"].unique():
        view = CDSView(source=cds, filters=[GroupFilter(column_name='group', group=x)])
        plot_figure.circle(x="x", y="y", name=str(x) + "_dot", source=cds, line_width=2, 
                           color=colors[color_count],
                           line_alpha=0.6, fill_alpha=0.6, size=12, muted_alpha=0, view=view)
        color_count += 1
    return plot_figure


def visuals(x, df, pca, column_names):
    """
    This function calls visualization functions.
    :param x: data on which dimensionality reduction has been performed
    :param df: dataframe with data before dimensionality reduction
    :param pca: the principal component analysis
    :param column_names: list with column names
    :return: plot_figure = bokehplot of the PCA
    """
    plot_figure = pca_visual(x, df, pca.explained_variance_ratio_.tolist())
    save(plot_figure)
    # visual_comp = component_visual(pca, column_names)
    # save(visual_comp)
    return plot_figure


def pca_clustering(df):
    """
    This function does a PCA on raw TIC data.
    :param df: a dataframe containing all samples for a PCA
    :return: reduced_data = data on which dimensionality reduction has been performed
    :return: new_df = dataframe with data before dimensionality reduction
    :return: pca = the principal component analysis
    :return: list(data.columns.values) = list with column names from data
    """
    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        # Retrieve the whisky sample TIC file location 
        loc = row['sample_tic_location']
        # open the TIC file using pickle and get the TIC intensity array
        ms = pickle.load(open(loc, 'rb'))
        ints = ms.intensity_array
        # Add the intensities to a temporary dataframe
        temp_df = pd.DataFrame(ints, columns=[index])
        temp_df = temp_df.transpose()
        # Add an extra column for the group name to the row of intensities.
        if len(row['sample_code'].split("_")) > 6:
            # This is to get the group name of a blend
            temp_df["group"] = row['sample_code'].split("_")[2] + "_" + row['sample_code'].split("_")[3].strip("1").strip("2").strip("3") + "_" + row['sample_code'].split("_")[4] + "_" + row['sample_code'].split("_")[5]
        else:
            # This is to get the group name of single malts 
            temp_df["group"] = row['sample_code'].split("_")[-2].strip("1").strip("2").strip("3").strip("4").strip("5")
        # Add an extra column for the sample name to the row of intensities. 
        temp_df["code"] = str(requests.get(f"{app_settings.DATABASE_URL}/whisky/?whisky_code={row['sample_code']}").json()[0]['whisky_name'])
        # Combine all the temporary data frames into a new one
        new_df = pd.concat([new_df, temp_df])
    # Fill all NaNs with 0
    new_df = new_df.fillna(0)
    # Get the data without sample group or name
    data = new_df.loc[:,~new_df.columns.isin(["group", "code"])]
    
    # Remove all columns with scan number below 700 or above 8000
    for idx in range(1,12988):
        if idx < 700 or idx > 8000:
            try:
                data = data.drop(columns=[idx])
            except:
                pass

    # scaled_data = StandardScaler().fit_transform(data)

    # Fit the model with the data and apply the dimensionality reduction on the data
    pca = PCA(n_components=2).fit(data)
    reduced_data = pca.transform(data)
    return reduced_data, new_df, pca, list(data.columns.values)


def main(whisky_id, sample_list):
    """
    The main calls other functions.
    :param whisky_id: id of the whisky that the user wants to compare other whiskies with
    :param sample_list: a list of whiskies that the user wants to compare to a whisky
    :return: plot_figure = bokehplot of the PCA
    """
    dict_list = []
    for s_id in sample_list + [whisky_id]:
        sample = requests.get(f"{app_settings.DATABASE_URL}/sample/?sample_id={s_id}").json()[0]
        dict_list.append(sample)
    tics_df = pd.DataFrame.from_dict(dict_list)
    x, df, pca, column_names = pca_clustering(tics_df)
    plot_figure = visuals(x, df, pca, column_names)
    return plot_figure
        

if __name__ == "__main__":
    visual_pca = main(1, list(range(2,64))) #zonder e = range(2,82) #met e = range(2,91) #AvsB = range(2,64)













