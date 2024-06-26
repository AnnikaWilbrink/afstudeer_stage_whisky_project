import requests
import json
import os
import pickle
from pathlib import Path
from django.shortcuts import render
from django.http import HttpResponseRedirect

from peak_comparator_aromas import compare_aroma_peaks
from peak_comparator_whisky import compare_whisky_peaks
from cluster_analysis import main

from bokeh.models import HoverTool, Legend, Circle
from bokeh.plotting import ColumnDataSource, figure
from bokeh.palettes import Category10
from bokeh.embed import components
import matplotlib.pyplot as plt
from config import Settings

app_settings = Settings()


def start_page(request):
    """
    This is the function for the homepage. This function gets the path
    of the static folder for the background image for the homepage.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    path = os.path.join(BASE_DIR, "static")
    return render(request, 'start_page.html', {
        'path': path,
    })


def aroma_page(request):
    """
    This is the function for the aromapage. This function gets all
    aromas from the database, capitalizes the aroma names, replaces
    the underscore in the aroma name with a space and gets the path
    of the static folder for the images of the aromas.
    """
    # Get all the aromas in the database
    aromas = requests.get(f"{app_settings.DATABASE_URL}/aromas").json()
    # Get the path of the static folder for the aroma images
    BASE_DIR = Path(__file__).resolve().parent.parent
    path = os.path.join(BASE_DIR, "static")
    # For each aroma, replace the underscore in the aroma name with a space,
    # capitalize the name and add each aroma to a list.
    aroma_list = []
    for dict in aromas:
        dict["name"] = dict["aroma_name"].replace(" ", "_")
        dict['aroma_name'] = dict['aroma_name'].capitalize()
        aroma_list.append(dict)
    return render(request, 'aroma_page.html', {
        'aroma_list': aroma_list, 
        'path': path,
    })


def aroma_info_page(request, aroma_id):
    """
    This is the function for the aroma infopage. This function gets
    all information of the aroma the user chose from the database and
    capitalizes the name of the aroma.
    """
    # Get the aroma from the database
    aroma_dict = requests.get(f"{app_settings.DATABASE_URL}/aroma/?aroma_id={aroma_id}").json()[0]
    # Capitalize the aroma name
    aroma_dict['aroma_name'] = aroma_dict['aroma_name'].capitalize()
    return render(request, 'aroma_info_page.html', {
        "aroma_dict": aroma_dict,
    })


def whisky_result_page(request):
    """
    This is the function for the whisky result page. This function 
    compares the clicked whisky with the checked whiskies.
    """
    # Get all the variables from the session
    whisky_id = request.session.get('whisky_id')
    sample_ids = request.session.get('sample_ids')
    whisky_code = request.session.get('whisky_code')
    sample_codes = request.session.get('sample_codes')
    script = request.session.get('script')
    div = request.session.get('div')

    # Use peak_comparator_whisky.py to compare the clicked whisky with all the checked whiskies
    df_whiskies = compare_whisky_peaks(whisky_id, sample_ids)
    # Get the whisky information from the database
    whisky_dict = requests.get(f"{app_settings.DATABASE_URL}/whisky/?whisky_code={whisky_code}").json()[0]
    whisky_list = []
    # Loop through all the sample codes and get the whisky information from those whiskies
    for idx in range(0, len(sample_codes)):
        dict = requests.get(f"{app_settings.DATABASE_URL}/whisky/?whisky_code={sample_codes[idx]}").json()[0]
        dict["similarity"] = int(df_whiskies[df_whiskies['sample_id'] == sample_ids[idx]]['perc_found'].to_list()[0])
        whisky_list.append(dict)
    # Sort the whisky list on the similarity
    sorted_list = sorted(whisky_list, key=lambda x: x['similarity'], reverse=True)
    # Add idexes to the sorted list
    for count, d in enumerate(sorted_list, start=1):
        d['count'] = count
    return render(request, 'whisky_result_page.html', {
        'whisky_dict': whisky_dict,
        'whisky_list': sorted_list,
        'script': script, 
        'div': div,
        })  


def plot_tics(whisky_id, df, whisky_name):
    """
    This function plots the TICs of the clicked whisky and the aromas.
    :param whisky_id: The id of the whisky
    :param df: A dataframe containing information about the aromas found in the whisky
    :param whisky_name: the name of the whisky
    :return: graph = the chromatogram
    """
    # Get the whisky from the database
    sample = requests.get(f"{app_settings.DATABASE_URL}/sample/?sample_id={whisky_id}").json()
    # Get TIC file name an open the TIC to get the mass spectrum
    filename = str(sample[0]['sample_tic_location'])
    ms = pickle.load(open(filename, 'rb'))
    # Get a list with retention time and a list with intensities from the MS
    tic_time = ms.time_list[0:ms.time_list.index(next(x for x in ms.time_list if x > 1830))]
    tic_inten = ms.intensity_array[0:len(tic_time)] 

    # Make a bokeh figure
    graph = figure(
        x_axis_label="Retention time",
        y_axis_label="Intensity",
        width=1150, 
        height=500, 
        sizing_mode='scale_width',
        tools='pan, save, reset, box_zoom'
    )
    graph.min_border_top = 10
    colors = Category10[10]

    # Plot the whisky TIC
    graph.line(tic_time, tic_inten,
        line_color = "blue",
        legend_label = whisky_name,
        line_width=2,
        line_alpha=0.5)

    # Loop through the aroma dataframe
    for index, row in df.iterrows():
        # Get the aroma from the database
        aroma = requests.get(f"{app_settings.DATABASE_URL}/aroma/?aroma_id={row['aroma_id']}").json()
        # Get the TIC file name and open the TIC to get the mass spectrum
        filename = str(aroma[0]['aroma_tic_location'])
        ms2 = pickle.load(open(filename, 'rb'))
        # Get a list with retention time and a list with intensities from the MS
        tic_time2 = ms2.time_list[0:ms2.time_list.index(next(x for x in ms2.time_list if x > 1830))]
        tic_inten2 = ms2.intensity_array[0:len(tic_time2)]
        
        # Make a ColumnDataSource with the data from the dataframe
        source = ColumnDataSource(data=dict(x1=row['s_rt'], y1=row['s_sum_intensities'], x2=row['a_rt'], y2=row['a_sum_intensities'],
                                  num=row['peak_num'], comp=row['compound'], color=colors[0:row['peak_num'][-1]]))
        # Plot the aroma TIC
        aroma_tic = graph.line(tic_time2, tic_inten2,
                                line_color = "red",
                                name="aroma",
                                legend_label = aroma[0]['aroma_name'].capitalize(),
                                line_width=2,
                                line_alpha=0.5)
        # Plot circles that indicate the matching aroma peak and whisky peak
        sample_peak = graph.circle(x='x1', y='y1', name="compounds", line_width=0.5, source=source, color='color',
                                  legend_label=aroma[0]['aroma_name'].capitalize(), line_alpha=0.6, fill_alpha=0.6, size=12)
        aroma_peak = graph.circle(x='x2', y='y2', name="compounds", line_width=0.5, source=source, color='color',
                                  legend_label=aroma[0]['aroma_name'].capitalize(), line_alpha=0.6, fill_alpha=0.6, size=12)
        # Make the aroma TIC and the circles invisible when the user opens the plot
        aroma_tic.visible = False
        sample_peak.visible = False
        aroma_peak.visible = False
        # Add a hover tool to the circles in the graph, which shows the compound of the 
        # corresponding peak on which the user is hovering
        graph.add_tools(HoverTool(renderers=[sample_peak, aroma_peak], tooltips="""
        <div>
            <div>
                <span style='font-size: 16px; color: #224499'>Peak:</span>
                <span style='font-size: 16px'>@num</span><br>
                <span style='font-size: 16px; color: #224499'>Compound:</span>
                <span style='font-size: 16px'>@comp</span>
            </div>
        </div>
        """))
    # The location of the legend is in the top right corner. If the user clicks an element in the legend,
    # the corresponding plot becomes visible or invisible.
    graph.legend.location = "top_right"
    graph.legend.click_policy="hide"
    return graph


def whisky_info_page(request, whisky_code):
    """
    This is the function for the whisky info page. This function 
    compares the clicked whisky with the aromas from the database
    and creates a graph. The clicked whisky can also be compared
    with other whiskies from the database. This function sends 
    the necessary information to the next page.
    """
    # Get the clicked whisky from the database
    whisky_id = int(requests.get(f"{app_settings.DATABASE_URL}/samples/{whisky_code}").json()[0]['sample_id'])
    whisky_dict = requests.get(f"{app_settings.DATABASE_URL}/whisky/?whisky_code={whisky_code}").json()[0]
    # Get all other whiskies from the database and save them in a list
    whisky_list = list(filter(lambda i: i['whisky_code'] != whisky_code, requests.get(f"{app_settings.DATABASE_URL}/whiskys").json()))

    # Replace the '_' with a '-' in the dates (format in the database for dates are dd_mm_yyyy)
    whisky_dict['whisky_distillation_date'] = whisky_dict['whisky_distillation_date'].replace("_", "-")
    whisky_dict['whisky_bottle_date'] = whisky_dict['whisky_bottle_date'].replace("_", "-")

    # If the compare button is clicked...
    if request.method == "POST":
        # Get all the checked whiskies
        checked_whiskies = request.POST.getlist('whiskies')
        # Get the sample ids from the checked whiskies
        checked_sample_ids = [int(requests.get(f"{app_settings.DATABASE_URL}/samples/{x}").json()[0]['sample_id']) for x in checked_whiskies]
        # Use cluster_analysis.py to make a cluster plot from all the checked whiskies that the user wants to compare with the clicked whisky
        plot_figure = main(whisky_id, checked_sample_ids)
        # This bokeh function returns two variables that can be used to display the cluster plot on an HTML page
        script, div = components(plot_figure)
        # With sessions, all these variables are sent to the function whisky_result_page() for the next page
        request.session['whisky_id'] = whisky_id
        request.session['sample_ids'] = checked_sample_ids
        request.session['whisky_code'] = whisky_code
        request.session['sample_codes'] = checked_whiskies
        request.session['script'] = script
        request.session['div'] = div
        # Redirect to the next page to show the compare results
        return HttpResponseRedirect('/compare_results/')
    # If the compare button isn't clicked... (This only happens when whisky_info_page first opens. This is to prevent these code lines to
    # run more than once)
    else:
        # Use peak_comparator_aroma.py to compare the clicked whisky with all the aromas in the database
        df_aromas, plot_path = compare_aroma_peaks(whisky_id)
        # Get a chromatogram of the clicked whisky and the aromas with the function plot_tics
        graph = plot_tics(whisky_id, df_aromas, whisky_dict['whisky_name'])
        # This bokeh function returns two variables that can be used to display the chromatogram on an HTML page
        script, div = components(graph)   
    return render(request, 'whisky_info_page.html', {
        "whisky_dict": whisky_dict,
        # "aroma_list": aroma_list,
        "whisky_list": whisky_list,
        "plot_path": plot_path,
        'script': script,
        'div': div,
    })


def whisky_page(request):
    """
    This is the function for the whiskypage. This function gets all
    the whiskies from the database.
    """
    whisky_list = requests.get(f"{app_settings.DATABASE_URL}/whiskys").json()
    return render(request, 'whisky_page.html', {
        'whisky_list': whisky_list,
        })    