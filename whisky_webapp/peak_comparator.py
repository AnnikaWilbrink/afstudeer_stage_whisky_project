import requests
import json
import pandas as pd
import numpy as np
import itertools
import pickle
from statistics import median, mean
from collections import defaultdict
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import seaborn
from bokeh.plotting import figure, output_file, show, save
import os
from bokeh.palettes import Turbo256, all_palettes, Cividis, turbo, viridis
import pyms.IonChromatogram
import pyms.GCMS
from pyms.GCMS.IO.ANDI import ANDI_reader


def plot_unique_peaks(unique_a, unique_b, file_a, file_b):
    graph = figure(width=1700, 
                   height=900)
    graph.xaxis.axis_label = "Retention Time (Rt)"
    graph.yaxis.axis_label = "Intensity (I)"
    file_dir = "/exports/nas/wilbrink.a/whisky_project/whisky_data/test_piek_aroma_tics/"
    colors = ["red", "blue"]
    count = 0
    for x in [file_a, file_b]:
        label = x.strip("training_whiskey_")
        ms = pickle.load(open(file_dir+x, 'rb'))
        tic_time = ms.time_list
        tic_inten = ms.intensity_array  
        graph.line(tic_time, tic_inten,
        line_color = colors[count],
        legend_label = label,
        line_width=2,
        line_alpha=0.5,
        muted_alpha=0)
        if x == file_a:
            graph.circle(x=unique_a["peak_retention_time"].to_list(), y=[sum(json.loads(x)) for x in unique_a["peak_intensities"].to_list()], name="unique_peaks_"+label, line_width=0.5, color=colors[0], 
                        legend_label="peaks_"+label, line_alpha=0.6, fill_alpha=0.6, size=15, muted_alpha=0)
        else:
            graph.circle(x=unique_b["peak_retention_time"].to_list(), y=[sum(json.loads(x)) for x in unique_b["peak_intensities"].to_list()], name="unique_peaks_"+label, line_width=0.5, color=colors[1], 
                            legend_label="peaks_"+label, line_alpha=0.6, fill_alpha=0.6, size=15, muted_alpha=0)
        count += 1
    
    graph.legend.location = "top_right"
    graph.legend.click_policy="mute"      
    output_file("/exports/nas/wilbrink.a/whisky_project/whisky_plotjes/A&Bunique_peaks.html")
    save(graph)


def plot_found_peaks(df, aroma_id):
    """

    """
    # print(df)
    graph = figure(width=1700, 
                   height=900)
    graph.xaxis.axis_label = "Retention Time (Rt)"
    graph.yaxis.axis_label = "Intensity (I)"

    colors = turbo(8)
    color_count = 1
    for i in range (1,6):
        aroma = requests.get(f"http://145.97.18.149:7899/aroma/?aroma_id={i}").json()
        aroma_peaks = requests.get(f"http://145.97.18.149:7899/peaks/?aroma_id={i}").json()
        a_peak_frame = pd.DataFrame(aroma_peaks)
        filename = str(aroma[0]['aroma_tic_location'])
        label = filename.split("/")[-1].strip("training_").strip("whiskey_")
        ms = pickle.load(open(filename, 'rb'))
        tic_time = ms.time_list
        tic_inten = ms.intensity_array
        graph.line(tic_time, tic_inten,
        line_color = colors[color_count],
        legend_label = label,
        line_width=2,
        line_alpha=0.5,
        muted_alpha=0)
        graph.circle(x=a_peak_frame["peak_retention_time"].to_list(), y=[sum(json.loads(x)) for x in a_peak_frame["peak_intensities"].to_list()], name=label, line_width=0.5, color=colors[color_count], 
                        legend_label=label, line_alpha=0.6, fill_alpha=0.6, size=15, muted_alpha=0)
        color_count+=1

    # for sample_id in np.unique(df["peak_sample_id"].to_list()).tolist():
    #     sample = requests.get(f"http://145.97.18.149:7899/sample/?sample_id={sample_id}").json()
    #     sample_peaks = requests.get(f"http://145.97.18.149:7899/peaks/?sample_id={sample_id}").json()
    #     s_peak_frame = pd.DataFrame(sample_peaks)
    #     filename = str(sample[0]['sample_tic_location'])
    #     # round(ms.intensity_list[x], 2)
    #     label = filename.split("/")[-1].strip("training_").strip("whiskey_")
    #     ms = pickle.load(open(filename, 'rb'))
    #     idx_hit = []
    #     tic_time = ms.time_list
    #     tic_inten = ms.intensity_array
    #     for rt in df[df["peak_sample_id"] == sample_id]["peak_s_rt"].to_list():
    #         idx_hit.append(tic_time.index(rt))
    #     for i in range(0, len(tic_inten)):
    #         if i not in idx_hit:
    #             tic_inten[i] = 0   
    #     graph.line(tic_time, tic_inten,
    #     line_color = colors[color_count],
    #     legend_label = label,
    #     line_width=2,
    #     line_alpha=0.5,
    #     muted_alpha=0)
    #     graph.circle(x=s_peak_frame["peak_retention_time"].to_list(), y=[sum(json.loads(x)) for x in s_peak_frame["peak_intensities"].to_list()], name="peaks_"+label, line_width=0.5, color=colors[color_count], 
    #              legend_label="peaks_"+label, line_alpha=0.6, fill_alpha=0.6, size=15, muted_alpha=0)
    #     color_count += 1

    graph.legend.location = "top_right"
    graph.legend.click_policy="mute"      
    output_file("/exports/nas/wilbrink.a/whisky_project/whisky_plotjes/PeaksA.html")
    save(graph)


def variance(data):
    """
    
    """
    mean = sum(data) / len(data)
    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / len(data)
    return variance    


def plot_perc_found_peaks(df):
    """

    """
    # print(df)
    for malt in df['single_malt'].unique():
        filename = "/exports/nas/wilbrink.a/whisky_project/whisky_plotjes/percFoundPeaks" + malt + ".png"
        blends = df[df['single_malt'] == malt]['blend'].to_list()
        perc_found = [int(perc) for perc in df[df['single_malt'] == malt]['perc_found'].to_list()]
        step_size = 5
        min_y = min(perc_found) -2
        if malt == "B1":
            step_size = (100 - min(perc_found))/5
            min_y = min(perc_found)
        
        plt.figure(figsize=(5,5))
        plt.plot(blends, perc_found) 
        plt.yticks(np.arange(min_y, 101, step_size))
        plt.ylabel('% found')
        plt.xlabel('blends')
        plt.suptitle("% found peaks from " + malt + " in blends")
        plt.savefig(filename)
    
    for malt in df['single_malt'].unique():
        filename = "/exports/nas/wilbrink.a/whisky_project/whisky_plotjes/percUniquePeaks" + malt + ".png"
        blends = df[df['single_malt'] == malt]['blend'].to_list()
        perc_unique = [int(perc) for perc in df[df['single_malt'] == malt]['perc_unique'].to_list()]
        # print(blends, perc_unique)
        plt.figure(figsize=(5,5))
        plt.plot(blends, perc_unique)
        plt.gca().set_ylim([min(perc_unique)-7, 105])
        plt.ylabel('% found')
        plt.xlabel('blends')
        plt.suptitle("% found unique peaks from " + malt + " in blends")
        plt.savefig(filename)


def plot_area_ratios(df):
    for malt in df['single_malt'].unique():
        blends = df[df['single_malt'] == malt]['blend'].to_list()        
        for blend in blends:
            filename = "/exports/nas/wilbrink.a/whisky_project/whisky_plotjes/areaRatio" + malt+"_"+blend  + ".png"
            plt.figure(figsize=(10,10))
            plt.ylabel('ratio')
            plt.xlabel('peak ids')
            ratios = json.loads(df[(df['single_malt'] == malt) & (df['blend'] == blend)]['area_ratios'].to_list()[0])
            peak_ids = json.loads(df[(df['single_malt'] == malt) & (df['blend'] == blend)]['peak_ids'].to_list()[0])
            plt.plot(peak_ids, ratios, label=blend)
            plt.ylim([-4,8])
            plt.yticks(np.arange(-4, 8, 1))
            plt.xticks(peak_ids)
            plt.suptitle("Area ratios from " + malt + " and " + blend)
            plt.grid()
            plt.legend()
            plt.savefig(filename)


def plot_medians(df, y):
    """

    """
    fig, ax = plt.subplots(1,1)
    seaborn.set(style = 'whitegrid')    
    seaborn.stripplot(x="blend", y=y, data=df, ax=ax, hue="single_malt")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    filename = y + ".png"
    plt.savefig(filename)


def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(13,7)) 
    plot = seaborn.heatmap(df, annot=True, ax=ax)
    # fig = plot.get_figure()
    plt.savefig("/exports/nas/wilbrink.a/whisky_project/whisky_plotjes/heatmap.png")


def visuals(hit_df, final_df, height_df):
    """

    """
    # plot_found_peaks(hit_df, 1)
    # plot_perc_found_peaks(final_df)
    plot_heatmap(height_df)
    # plot_area_ratios(final_df)
    # plot_medians(final_df, "median_rt")
    # plot_medians(final_df, "median_area")
    # plot_medians(final_df, "median_corr")
    # plot_medians(final_df, "variance_rt")
    # plot_medians(final_df, "variance_area")


def check_best_match(df):
    """

    """
    # Create a new df for the hits
    hit_df = pd.DataFrame()
    # Create a list with the unique sample ids
    sample_ids = np.unique(df["s_peak_id"].to_list()).tolist()
    # For each unique sample id
    for id in sample_ids:
        # If rt_diff is less than 6, sort the rows on rt_diff and get the first row (the hit with the lowest rt_diff)
        # print(df[(df['s_peak_id'] == id) & (df['rt_diff'] < 6)].sort_values(by=['rt_diff']))
        best_hit = df[(df['s_peak_id'] == id) & (df['rt_diff'] < 6)].sort_values(by=['rt_diff']).head(1)
        # Check if the row and new hit_df is not empty
        if best_hit.empty == False and hit_df.empty == False:
            # Check if the aroma_peak_id from the best hit is already in the dataframe
            if best_hit['a_peak_id'].values[0] in hit_df['a_peak_id'].values:
                # Check if the difference in retention time from the hit in the df is greater than 
                # the difference in retention time from the new hit. In that case get the index of 
                # the old hit, drop the old hit from the df and add the new hit.
                if hit_df.loc[hit_df['a_peak_id'] == best_hit['a_peak_id'].values[0]]['rt_diff'].values[0] > best_hit['rt_diff'].values[0]:
                    idx = hit_df.loc[hit_df['a_peak_id'] == best_hit['a_peak_id'].values[0]].index.values.astype(int)[0]
                    hit_df.drop(idx, axis=0, inplace=True)
                    hit_df = pd.concat([hit_df, df[(df['s_peak_id'] == id) & (df['rt_diff'] < 6)].sort_values(by=['rt_diff']).head(1)])
            else:
                hit_df = pd.concat([hit_df, df[(df['s_peak_id'] == id) & (df['rt_diff'] < 6)].sort_values(by=['rt_diff']).head(1)])
        else:           
            hit_df = pd.concat([hit_df, df[(df['s_peak_id'] == id) & (df['rt_diff'] < 6)].sort_values(by=['rt_diff']).head(1)])
    return hit_df


def check_aroma_peak_order(aroma_id):
    """

    """
    peaks = requests.get(f"http://145.97.18.149:7899/total_peaks/?sample_id={aroma_id}").json()
    peak_ids = []
    for peak in peaks:
        peak_ids.append(peak['peak_id']) 
    return peak_ids


def check_peak_order(matchframe, sample_id, aroma_id, final_df):
    """
    
    """
    # Get a list with the ids from all the aroma peaks
    aroma_peak_ids = check_aroma_peak_order(aroma_id)
    # Filter the dataframe with matches so that one sample peak has one hit with an aroma peak
    hit_df = check_best_match(matchframe)
    median_rt = median(hit_df['rt_diff'].to_list())
    median_area = median(hit_df['area_ratio'].to_list())
    median_corr = median(hit_df['correlation'].to_list())
    variance_rt = variance(hit_df['rt_diff'].to_list())
    variance_area = variance(hit_df['area_ratio'].to_list())
    # print(median_rt, median_area, median_corr, variance_rt, variance_area)
    singlemalts = ["", "A1", "B1", "C1", "D1", "E1", "AB1_10_90", "AB1_20_80", "AB1_50_50", "AB1_80_20", "AB1_90_10"]


    final_df = pd.concat([final_df, pd.DataFrame([{'sample_id': sample_id, 'aroma_id': aroma_id, 
                                                   'single_malt': singlemalts[aroma_id], 'blend': singlemalts[sample_id],
                                                   'median_rt': median_rt, 'median_area': median_area,
                                                   'median_corr': median_corr, 'variance_rt': variance_rt,
                                                   'variance_area': variance_area,
                                                   'perc_found': str(round(len(hit_df)/len(aroma_peak_ids)*100)),
                                                   }])], ignore_index=True)
    #print(hit_df)
    # print("Aantal pieken in de aroma:", len(aroma_peak_ids))
    # print("Aantal teruggevonden pieken:", len(hit_df))
    # print("Procent teruggevonden pieken:", str(round(len(hit_df)/len(aroma_peak_ids)*100)) +"%")
    return final_df, hit_df


def calculate_correlation(aroma_peak, sample_peak, matchesframe):
    """

    """
    highest_num = 0
    hit=0
    peak_matches=[]
    query_matches=[]
    s_masses = json.loads(sample_peak['peak_masses'])
    s_intensities = json.loads(sample_peak['peak_intensities'])
    a_masses = json.loads(aroma_peak['peak_masses'])
    a_intensities = json.loads(aroma_peak['peak_intensities'])
    for ms, ma in itertools.product(s_masses,a_masses):
        if abs(ma-ms) <= 0.5:
            hit+=1
            peak_matches.append(s_intensities[s_masses.index(ms)])
            query_matches.append(a_intensities[a_masses.index(ma)])
    if hit >= 1:
        product = sum([x*y for x,y in zip(peak_matches, query_matches)])
        corr = np.power(product / (norm(s_intensities)*norm(a_intensities)),2)
  
    if corr >= 0.7:
        matchesframe = pd.concat([matchesframe, pd.DataFrame([{'s_peak_id': sample_peak['peak_id'], 'peak_sample_id': sample_peak['peak_sample_id'],
                                                                'peak_s_rt': sample_peak['peak_retention_time'], 'peak_s_area': sample_peak['peak_area'], 'peak_s_height': sample_peak['peak_height'],
                                                                'a_peak_id': aroma_peak['peak_id'], 'peak_aroma_id': aroma_peak['peak_sample_id'],
                                                                'peak_a_rt': aroma_peak['peak_retention_time'], 'peak_a_area': aroma_peak['peak_area'], 'peak_a_height': aroma_peak['peak_height'],
                                                                'area_ratio': aroma_peak['peak_area'] / sample_peak['peak_area'], 
                                                                'rt_diff': abs(sample_peak['peak_retention_time'] - aroma_peak['peak_retention_time']), 'correlation':corr                                                            
                                                                }])], ignore_index=True)
    return matchesframe
    

def filter_peaks(matchesframe, s_id, a_id):
    """

    """
    # Get peaks from database
    peaks_sample1 = requests.get(f"http://145.97.18.149:7899/peaks/?sample_id={a_id}").json()
    peaks_sample2 = requests.get(f"http://145.97.18.149:7899/peaks/?sample_id={s_id}").json()
    # Add splashkey components 2+3 to the sample and aroma dictionaries
    for s2_peak in peaks_sample2:
        s2_peak['peak_splash_2_and_3'] = "-".join(s2_peak['peak_splash'].split("-")[1:3])
        s2_peak['peak_splash_2'] = s2_peak['peak_splash'].split("-")[1]
        s2_peak['peak_splash_3'] = s2_peak['peak_splash'].split("-")[2]  
    for s1_peak in peaks_sample1:
        s1_peak['peak_splash_2_and_3'] = "-".join(s1_peak['peak_splash'].split("-")[1:3])
        s1_peak['peak_splash_2'] = s1_peak['peak_splash'].split("-")[1]
        s1_peak['peak_splash_3'] = s1_peak['peak_splash'].split("-")[2]
        s1_splash2 = np.base_repr(int(str(s1_peak['peak_splash_2']),36),base=3).rjust(10,"0")
        for s2_peak in peaks_sample2:
            # Calculate the correlation between the peaks if their splashkey components 2+3 match         
            s2_splash2 = np.base_repr(int(str(s2_peak['peak_splash_2']),36),base=3).rjust(10,"0")
            diff_splash2 = sum([abs(int(a)-int(b)) for a,b in zip(s1_splash2,s2_splash2)])
            diff_splash3 = sum([abs(int(a)-int(b)) for a,b in zip(str(s1_peak['peak_splash_3']),str(s2_peak['peak_splash_3']))])
            if diff_splash2 <= 4 and diff_splash3 <= 4:
                matchesframe = calculate_correlation(s1_peak, s2_peak, matchesframe)
    return matchesframe            


def compare_whisky_a_and_b():
    """

    """
    matchesframe = pd.DataFrame()
    final_df = pd.DataFrame()
    matchesframe = filter_peaks(matchesframe, 1, 2)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    sortedframe = matchesframe.sort_values(by=['peak_sample_id', 'peak_s_rt', 'correlation'])
    final_df, hit_df = check_peak_order(sortedframe.reset_index(), 1, 2, final_df)
    # print(hit_df)
    unique_peaks_a = pd.DataFrame(requests.get(f"http://145.97.18.149:7899/peaks/?sample_id={1}").json())
    unique_peaks_b = pd.DataFrame(requests.get(f"http://145.97.18.149:7899/peaks/?sample_id={2}").json())
    unique_peaks_a = unique_peaks_a[~unique_peaks_a["peak_id"].isin(hit_df['s_peak_id'].to_list())]
    unique_peaks_b = unique_peaks_b[~unique_peaks_b["peak_id"].isin(hit_df['a_peak_id'].to_list())]
    # plot_unique_peaks(unique_peaks_a, unique_peaks_b, "training_whiskey_A1_1", "training_whiskey_B1_1")
    print(unique_peaks_a)
    return unique_peaks_a, unique_peaks_b


def compare_malt_and_blend():
    """

    """
    unique_a, unique_b = compare_whisky_a_and_b()
    matchesframe = pd.DataFrame()
    for sample_id in range(6,11):
        for aroma_id in range(1,3):
            matchesframe = filter_peaks(matchesframe, sample_id, aroma_id)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    sortedframe = matchesframe.sort_values(by=['peak_sample_id', 'peak_s_rt', 'correlation'])
    # print(sortedframe)
    final_df = pd.DataFrame()
    hit_df = pd.DataFrame()
    for sample_id in range(6, int(sortedframe.max()['peak_sample_id'])+1):
        matchdf = sortedframe.loc[sortedframe['peak_sample_id'] == sample_id]
        for aroma_id in range(1,int(sortedframe.max()['peak_aroma_id'])+1):
            matchdf2 = matchdf.loc[matchdf['peak_aroma_id'] == aroma_id]
            final_df, df = check_peak_order(matchdf2.reset_index(), sample_id, aroma_id, final_df)
            hit_df = pd.concat([hit_df, df])
    # print(hit_df)
    # print(unique_a)
    # print(unique_b['peak_id'].to_list())
    #visuals(hit_df, final_df)
    # print(hit_df[hit_df['peak_aroma_id'] == 1])
    singlemalts = ["", "A1", "B1", "C1", "D1", "E1"]
    blends = ["", "", "", "","","", "AB1_10_90", "AB1_20_80", "AB1_50_50", "AB1_80_20", "AB1_90_10"]
    df_array = []
 
    df_row = []
    df_row.append(singlemalts[2])
    for id in unique_a['peak_id'].to_list() + unique_b['peak_id'].to_list():
        if id in unique_b['peak_id'].to_list():
            idx = unique_b['peak_id'].to_list().index(id)
            df_row.append(unique_b['peak_height'].to_list()[idx])
        else:
            df_row.append(0)
    df_array.append(df_row)
    
    for single_malt in hit_df['peak_aroma_id'].unique():
        
        for blend in hit_df[hit_df['peak_aroma_id'] == single_malt]['peak_sample_id'].unique():
            df_row = []
            if single_malt == 1:
                unique_in_blend = hit_df[(hit_df['peak_aroma_id'] == single_malt) & (hit_df['peak_sample_id'] == blend) & (hit_df['a_peak_id'].isin(unique_a['peak_id'].to_list()))]
                print("Aantal unieke pieken in "+ singlemalts[single_malt] +":", len(unique_a['peak_id'].to_list()))
                print("Aantal teruggevonden unieke pieken in " + blends[blend] + ":", len(unique_in_blend), unique_in_blend['a_peak_id'].to_list())
                print("%:", round(len(unique_in_blend)/len(unique_a['peak_id'].to_list())*100))
                print(unique_in_blend[['a_peak_id', 'peak_a_height', 'peak_s_height']])
                print()
                final_df.loc[(final_df['sample_id'] == blend) & (final_df['aroma_id'] == single_malt), "perc_unique"] = str(round(len(unique_in_blend)/len(unique_a['peak_id'].to_list())*100))

                df_row.append(blends[blend])
                for id in unique_a['peak_id'].to_list():
                    if id in unique_in_blend['a_peak_id'].to_list():
                        idx = unique_in_blend['a_peak_id'].to_list().index(id)
                        df_row.append(unique_in_blend['peak_s_height'].to_list()[idx])
                    else:
                        df_row.append(0)
                df_array.append(df_row)
            else:
                unique_in_blend = hit_df[(hit_df['peak_aroma_id'] == single_malt) & (hit_df['peak_sample_id'] == blend) & (hit_df['a_peak_id'].isin(unique_b['peak_id'].to_list()))]
                print("Aantal unieke pieken in "+ singlemalts[single_malt] +":", len(unique_b['peak_id'].to_list()))
                print("Aantal teruggevonden unieke pieken in " + blends[blend] + ":", len(unique_in_blend), unique_in_blend['a_peak_id'].to_list())
                print("%:", round(len(unique_in_blend)/len(unique_b['peak_id'].to_list())*100))
                print(unique_in_blend[['a_peak_id', 'peak_a_height', 'peak_s_height', 'peak_a_height']])
                print()
                final_df.loc[(final_df['sample_id'] == blend) & (final_df['aroma_id'] == single_malt), "perc_unique"] = str(round(len(unique_in_blend)/len(unique_b['peak_id'].to_list())*100))
                
                df_row.append(blends[blend])
                for id in unique_b['peak_id'].to_list():
                    if id in unique_in_blend['a_peak_id'].to_list():
                        idx = unique_in_blend['a_peak_id'].to_list().index(id)
                        df_row.append(unique_in_blend['peak_s_height'].to_list()[idx])
                    else:
                        df_row.append(0)
                df_array.append(df_row)

            final_df.loc[(final_df['sample_id'] == blend) & (final_df['aroma_id'] == single_malt), "area_ratios"] = str(unique_in_blend['area_ratio'].to_list())
            final_df.loc[(final_df['sample_id'] == blend) & (final_df['aroma_id'] == single_malt), "peak_ids"] = str(unique_in_blend['a_peak_id'].to_list())
            
        if single_malt == 1:
            df_row = []
            df_row.append(singlemalts[single_malt])
            for id in unique_a['peak_id'].to_list() + unique_b['peak_id'].to_list():
                if id in unique_a['peak_id'].to_list():
                    idx = unique_a['peak_id'].to_list().index(id)
                    df_row.append(unique_a['peak_height'].to_list()[idx])
                else:
                    df_row.append(0)
                
            df_array.append(df_row)
    d = {}
    for items in df_array:
        if items[0] not in d:
            d[items[0]] = [items[0]]
        d[items[0]] += items[1:] 
    df_array = list(d.values())

    unique_peaks_df = pd.DataFrame([x[1:] for x in df_array], index=[x[0] for x in df_array], columns=unique_a['peak_id'].to_list() + unique_b['peak_id'].to_list())
    visuals(hit_df, final_df, unique_peaks_df)


def compare_aroma_and_whisky():
    """

    """
    matchesframe = pd.DataFrame()
    for sample_id in range(1,6):
        for aroma_id in range(6,11):
            matchesframe = filter_peaks(matchesframe, sample_id, aroma_id, "aroma")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    sortedframe = matchesframe.sort_values(by=['peak_sample_id', 'peak_s_rt', 'correlation'])
    
    final_df = pd.DataFrame()
    hit_df = pd.DataFrame()
    for sample_id in range(1, int(sortedframe.max()['peak_sample_id'])+1):
        matchdf = sortedframe.loc[sortedframe['peak_sample_id'] == sample_id]
        for aroma_id in range(1,int(sortedframe.max()['peak_aroma_id'])+1):
            matchdf2 = matchdf.loc[matchdf['peak_aroma_id'] == aroma_id]
            final_df, df = check_peak_order(matchdf2.reset_index(), sample_id, aroma_id, final_df)
            hit_df = pd.concat([hit_df, df])
    visuals(hit_df, final_df)


def main():
    """

    """
    # compare_aroma_and_whisky()
    compare_malt_and_blend()
    
    


if __name__ == "__main__":
    main()
