import requests
import json
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
from config import Settings

app_settings = Settings()


def plot_aromas(df):
    """
    This function makes a barplot with the percentage of aroma in
    the whisky and the percentage of matching compounds.
    :param df: a dataframe with the most relevant information about 
     the matching aromas
    :return: plot_path = the path where the bar plot is saved
    """
    df = df.sort_values(by=['perc_aroma_in_whisky'], ascending=False)
    fig, ax = plt.subplots(figsize=(11,5.5))
    ax2 = ax.twinx()
    aroma_ids = df['aroma_id'].to_list()
    aromas = []
    for id in aroma_ids:
       aroma_entry = requests.get(f"{app_settings.DATABASE_URL}/aroma/?aroma_id={id}").json()[0]
       aromas.append(aroma_entry["aroma_name"].replace(" ", "\n"))
    y = [float(x) for x in df['perc_aroma_in_whisky'].to_list()]
    y2 = [int(x) for x in df['perc_peaks_found'].to_list()]
    x = np.arange(len(aromas))
    width = 0.2
    ax.bar(x-(width/2), y, color="red", width=width, label="Aromas found", alpha = 0.5)
    ax2.bar(x+(width/2), y2, color="blue", width=width, label="Aroma compounds found", alpha = 0.5)
    ax.set_ylabel('% aroma')
    ax2.set_ylabel('% matching compounds')
    ax.set_xticks(x)
    ax.set_xticklabels(aromas, fontsize='small')

    BASE_DIR = Path(__file__).resolve().parent
    plot_path = os.path.join(BASE_DIR, "static") + "/media/barplot.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path



def calc_perc_aroma_in_whisky(df, sample_id):
    """
    This function calculates how much of the aroma is
    present in the whisky and adds it to a dataframe.
    :param df: a dataframe with the most relevant information about 
     the matching aromas
    :param sample_id: id of the whisky
    :return: df = a dataframe with the most relevant information about 
     the matching aromas
    """
    sample_peaks = requests.get(f"{app_settings.DATABASE_URL}/peaks/?sample_id={sample_id}").json()
    total_height = sum([peak['peak_height'] for peak in sample_peaks])
    df['perc_aroma_in_whisky'] = [round((height/total_height)*100,1) for height in df['sum_height'].tolist()] 
    return df


def check_best_match(df):
    """
    This function gets the best match between peaks if a peak
    has multiple matches.
    :param df: dataframe with peakmatches
    :return: hit_df = the new dataframe with matches where each whisky peak 
     has only one hit with an aroma peak
    """
    # Create a new df for the hits and a temporary df
    hit_df = pd.DataFrame()
    temp_df = pd.DataFrame()
    # Create a list with the unique sample peak ids
    sample_ids = np.unique(df["s_peak_id"].to_list()).tolist()
    # Create a list with the unique aroma peak ids
    aroma_ids = np.unique(df["a_peak_id"].to_list()).tolist()
    # To filter out the multiple matches for the sample peaks, loop through all the unique sample peak ids 
    for id in sample_ids:
        # Get all the hits from that sample id
        hit = df[(df['s_peak_id'] == id) & (df['rt_diff'] <= 20)]
        # If there is only one hit, add that hit to temp_df
        if len(hit) == 1: 
            temp_df = pd.concat([temp_df, hit])
        # If there are more hits, add the hit with the most compound matches and the lowest retention time difference to temp_df           
        elif len(hit) > 1: 
            temp_df = pd.concat([temp_df, hit.sort_values(['len_matches', 'rt_diff'], ascending=[False, True]).head(1)])
    if temp_df.empty == False:
        # To filter out the multiple matches for the aroma peaks, loop through all the unique aroma peak ids and
        # add the hit with the highest correlation and the lowest retention time difference to hit_df
        for id in aroma_ids:
            hit_df = pd.concat([hit_df, temp_df[temp_df['a_peak_id'] == id].sort_values(['correlation', 'rt_diff'], ascending=[False, True]).head(1)])
    return hit_df


def get_aroma_peak_list(aroma_id):
    """
    This function makes a list where the ids of all the detected 
    aroma peaks are stored in.
    :param aroma_id: id of the aroma
    :return: peak_ids = list of peak ids 
    """
    peaks = requests.get(f"{app_settings.DATABASE_URL}/total_peaks/?aroma_id={aroma_id}").json()
    peak_ids = []
    for peak in peaks:
        peak_ids.append(peak['peak_id']) 
    return peak_ids


def check_peak_order(matchframe, sample_id, aroma_id, final_df):
    """
    This function creates a dataframe to store the whisky id, aroma id
    and relevant information about the matching aroma.
    :param matchframe: a dataframe with the information of matching peaks
    :param sample_id: id of the whisky
    :param aroma_id: id of the aroma
    :param final_df: a dataframe with the most relevant information about 
     the matching aroma
    :return: final_df = a dataframe with the most relevant information about 
     the matching aroma
    :return: hit_df = a dataframe with the information of matching peaks
    """
    # Get a list with the ids from all the aroma peaks
    aroma_peak_ids = get_aroma_peak_list(aroma_id)
    # Filter the dataframe with matches so that one sample peak has one hit with an aroma peak and vice versa
    hit_df = check_best_match(matchframe)
    
    if len(aroma_peak_ids) != 0 and len(hit_df) != 0:
        final_df = pd.concat([final_df, pd.DataFrame([{'sample_id': sample_id, 'aroma_id': aroma_id, 's_sum_intensities': hit_df['s_sum_intensities'].to_list(),
                                                       's_rt': hit_df['peak_s_rt'].to_list(), 'a_sum_intensities': hit_df['a_sum_intensities'].to_list(),
                                                       'a_rt': hit_df['peak_a_rt'].to_list(), 'compound': hit_df['best_match'].to_list(), 
                                                       'sum_height': sum(hit_df['peak_s_height'].to_list()), 'peak_num': list(range(1, len(hit_df['best_match'].to_list())+1)),
                                                       'perc_peaks_found': str(round(len(hit_df)/len(aroma_peak_ids)*100))
                                                       }])], ignore_index=True)
        print(aroma_id)
        print("Number of peaks in aroma:", len(aroma_peak_ids))
        print("Number of matching peaks:", len(hit_df))
        print("Percentage of matching peaks:", str(round(len(hit_df)/len(aroma_peak_ids)*100)) +"%")
        print()
        
    return final_df, hit_df


def calculate_correlation(aroma_peak, sample_peak, matchesframe):
    """
    This function calculates the correlation between a whisky peak and 
    an aroma peak and stores the peak information in a dataframe.
    :param aroma_peak: peak information from the database of the aroma
    :param sample_peak: peak information from the database of the whisky 
    :param matchesframe: a dataframe with the information of matching peaks
    :return: matchesframe = a dataframe with the information of matching peaks
    """
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
        compounds_sample = json.loads(sample_peak['peak_compound'])
        compounds_aroma = json.loads(aroma_peak['peak_compound'])
        if compounds_aroma[0] in compounds_sample:
    
            matchesframe = pd.concat([matchesframe, pd.DataFrame([{'s_peak_id': sample_peak['peak_id'], 'peak_sample_id': sample_peak['peak_sample_id'],
                                                                'peak_s_rt': sample_peak['peak_retention_time'], 'peak_s_height': sample_peak['peak_height'], 
                                                                'a_peak_id': aroma_peak['peak_id'], 'peak_aroma_id': aroma_peak['peak_aroma_id'],
                                                                'peak_a_rt': aroma_peak['peak_retention_time'],  'peak_a_height': aroma_peak['peak_height'],
                                                                's_sum_intensities': sum(s_intensities), 'a_sum_intensities': sum(a_intensities),
                                                                'rt_diff': abs(sample_peak['peak_retention_time'] - aroma_peak['peak_retention_time']), 
                                                                'compounds_sample': compounds_sample, 'compounds_aroma': compounds_aroma, 
                                                                'len_matches': len([s for s in compounds_sample for a in compounds_aroma if s == a]),
                                                                'best_match': [a for a in compounds_aroma for s in compounds_sample if a == s][0], #[s for s in compounds_sample for a in compounds_aroma if s == a][0],
                                                                'correlation':corr                                                            
                                                                }])], ignore_index=True)
    return matchesframe


def filter_peaks(matchesframe, sample_id, aroma_id):
    """
    This function retrieves the peak information from the database and
    uses the splash keys to filter the peaks so that only peaks are
    compared with similar mass spectra.
    :param matchesframe: a dataframe with the information of matching peaks
    :param sample_id: id of the whisky that the user wants to the aromas of
    :param aroma_id: id of the aroma
    :return: matchesframe = a dataframe with the information of matching peaks
    """
    # Get peaks from database
    peaks_sample1 = requests.get(f"http://145.97.18.149:7899/peaks/?aroma_id={aroma_id}").json()
    peaks_sample2 = requests.get(f"http://145.97.18.149:7899/peaks/?sample_id={sample_id}").json()
    # Add splashkey components 2+3 to the sample and aroma dictionaries
    for s2_peak in peaks_sample2:
        s2_peak['peak_splash_2_and_3'] = "-".join(s2_peak['peak_splash'].split("-")[1:3])
        s2_peak['peak_splash_2'] = s2_peak['peak_splash'].split("-")[1]
        s2_peak['peak_splash_3'] = s2_peak['peak_splash'].split("-")[2]  
    for s1_peak in peaks_sample1:
        s1_peak['peak_splash_2_and_3'] = "-".join(s1_peak['peak_splash'].split("-")[1:3])
        s1_peak['peak_splash_2'] = s1_peak['peak_splash'].split("-")[1]
        s1_peak['peak_splash_3'] = s1_peak['peak_splash'].split("-")[2]
        # Change the splash key component 2 from base-36 to base-3
        # https://www.unitconverters.net/numbers/base-36-to-base-3.htm
        s1_splash2 = np.base_repr(int(str(s1_peak['peak_splash_2']),36),base=3).rjust(10,"0")
        for s2_peak in peaks_sample2:
            # Change the splash key component 2 from base-36 to base-3
            # https://www.unitconverters.net/numbers/base-36-to-base-3.htm         
            s2_splash2 = np.base_repr(int(str(s2_peak['peak_splash_2']),36),base=3).rjust(10,"0")
            # Calculate the difference between the splash key components
            diff_splash2 = sum([abs(int(a)-int(b)) for a,b in zip(s1_splash2,s2_splash2)])
            diff_splash3 = sum([abs(int(a)-int(b)) for a,b in zip(str(s1_peak['peak_splash_3']),str(s2_peak['peak_splash_3']))])
            # If the difference between components is less than or equal to 4, calculate the
            # correlation between the peak mass spectra.
            if diff_splash2 <= 4 and diff_splash3 <= 4:
                matchesframe = calculate_correlation(s1_peak, s2_peak, matchesframe)
    return matchesframe


def compare_aroma_peaks(sample_id):
    """
    This function acts as the main and calls other functions.
    :param sample_id: id of the whisky that the user wants to know the aromas of
    :return: final_df = a dataframe with the most relevant information about 
     the matching aromas
    :return: plot_path = the path where the bar plot is saved
    """
    matchesframe = pd.DataFrame()
    for aroma_id in range(1,55):
        matchesframe = filter_peaks(matchesframe, sample_id, aroma_id)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Sort the dataframe by id, retention time and correlation
    sortedframe = matchesframe.sort_values(by=['peak_sample_id', 'peak_s_rt', 'correlation'])
    
    final_df = pd.DataFrame()
    hit_df = pd.DataFrame()
    for aroma_id in range(1,int(sortedframe.max()['peak_aroma_id'])+1):
        matchdf = sortedframe.loc[sortedframe['peak_aroma_id'] == aroma_id]
        final_df, df = check_peak_order(matchdf.reset_index(), sample_id, aroma_id, final_df)
        hit_df = pd.concat([hit_df, df])
    final_df = calc_perc_aroma_in_whisky(final_df, sample_id)
    print(final_df)
    plot_path = plot_aromas(final_df)
    return final_df, plot_path


if __name__ == "__main__":
    df, plot_path = compare_aroma_peaks(46)
