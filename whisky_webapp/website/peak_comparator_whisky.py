import requests
import json
import pandas as pd
import numpy as np
import itertools
from numpy.linalg import norm
from config import Settings

app_settings = Settings()


def check_best_match(df):
    """
    This function gets the best match between peaks if a peak
    has multiple matches.
    :param df: dataframe with peakmatches
    :return: hit_df = the new dataframe with matches where each whisky peak 
     has only one hit with a sample peak
    """
    # Create a new df for the hits and a temporary df
    hit_df = pd.DataFrame()
    temp_df = pd.DataFrame()
    # Create a list with the unique whisky peak ids
    whisky_ids = np.unique(df["w_peak_id"].to_list()).tolist()
    # Create a list with the unique sample peak ids
    sample_ids = np.unique(df["s_peak_id"].to_list()).tolist()
    # To filter out the multiple matches for the whisky peaks, loop through all the unique whisky peak ids 
    for id in whisky_ids:
        # Get all the hits from that whisky id
        hit = df[(df['w_peak_id'] == id)  & (df['rt_diff'] <= 20)]
        # If there is only one hit, add that hit to temp_df
        if len(hit) == 1: 
            temp_df = pd.concat([temp_df, hit])
        # If there are more hits, add the hit with the most compound matches and the lowest retention time difference to temp_df           
        elif len(hit) > 1: 
            temp_df = pd.concat([temp_df, hit.sort_values(['len_matches', 'rt_diff'], ascending=[False, True]).head(1)])
    if temp_df.empty == False:
        # To filter out the multiple matches for the sample peaks, loop through all the unique sample peak ids and
        # add the hit with the highest correlation and the lowest retention time difference to hit_df
        for id in sample_ids:
            hit_df = pd.concat([hit_df, temp_df[temp_df['s_peak_id'] == id].sort_values(['correlation', 'rt_diff'], ascending=[False, True]).head(1)])
    return hit_df


def get_sample_peak_list(sample_id):
    """
    This function makes a list where the ids of all the detected 
    whisky peaks are stored in.
    :param sample_id: id of the whisky that the user wants to compare to a whisky
    :return: peak_ids = list of peak ids 
    """
    peaks = requests.get(f"{app_settings.DATABASE_URL}/total_peaks/?sample_id={sample_id}").json()
    peak_ids = []
    for peak in peaks:
        peak_ids.append(peak['peak_id']) 
    return peak_ids


def check_peak_order(matchframe, whisky_id, sample_id, final_df):
    """
    This function creates a dataframe to store the ids of the whiskies and
    the percentage of matching peaks between them.
    :param matchframe: a dataframe with the information of matching peaks
    :param whisky_id: id of the whisky that the user wants to compare other whiskies with
    :param sample_id: id of the whisky that the user wants to compare to a whisky
    :param final_df: a dataframe with the percentage of matching peaks between whiskies
    :return: final_df = a dataframe with the percentage of matching peaks between whiskies
    :return: hit_df = a dataframe with the information of matching peaks
    """
    # Get a list with the ids from all the sample peaks
    sample_peak_ids = get_sample_peak_list(sample_id)
    # Filter the dataframe with matches so that one whisky peak has one hit with a sample peak
    hit_df = check_best_match(matchframe)

    final_df = pd.concat([final_df, pd.DataFrame([{'whisky_id': whisky_id, 'sample_id': sample_id, 
                                                   'perc_found': str(round(len(hit_df)/len(sample_peak_ids)*100)),
                                                   }])], ignore_index=True)

    print("Number of peaks in sample "+ str(sample_id) +":", len(sample_peak_ids))
    print("Number of matching peaks:", len(hit_df))
    print("Percentage of matching peaks:", str(round(len(hit_df)/len(sample_peak_ids)*100)) +"%")
    print()
    return final_df, hit_df


def calculate_correlation(sample_peak, whisky_peak, matchesframe):
    """
    This function calculates the correlation between two whisky peaks
    and stores the peak information in a dataframe.
    :param sample_peak: peak information from the database of the whisky
     that the user wants to compare to a whisky
    :param whisky_peak: peak information from the database of the whisky 
     that the user wants to compare whiskies to
    :param matchesframe: a dataframe with the information of matching peaks
    :return: matchesframe = a dataframe with the information of matching peaks
    """
    hit=0
    peak_matches=[]
    query_matches=[]
    w_masses = json.loads(whisky_peak['peak_masses'])
    w_intensities = json.loads(whisky_peak['peak_intensities'])
    s_masses = json.loads(sample_peak['peak_masses'])
    s_intensities = json.loads(sample_peak['peak_intensities'])
    for mw, ms in itertools.product(w_masses,s_masses):
        if abs(ms-mw) <= 0.5:
            hit+=1
            peak_matches.append(w_intensities[w_masses.index(mw)])
            query_matches.append(s_intensities[s_masses.index(ms)])
    if hit >= 1:
        product = sum([x*y for x,y in zip(peak_matches, query_matches)])
        corr = np.power(product / (norm(w_intensities)*norm(s_intensities)),2)
  
    if corr >= 0.7:
        compounds_whisky = json.loads(whisky_peak['peak_compound'])
        compounds_sample = json.loads(sample_peak['peak_compound'])
        if compounds_whisky[0] in compounds_sample:
            matchesframe = pd.concat([matchesframe, pd.DataFrame([{'w_peak_id': whisky_peak['peak_id'], 'peak_whisky_id': whisky_peak['peak_sample_id'],
                                                                    'peak_w_rt': whisky_peak['peak_retention_time'], 's_peak_id': sample_peak['peak_id'], 
                                                                    'peak_sample_id': sample_peak['peak_sample_id'], 'peak_s_rt': sample_peak['peak_retention_time'], 
                                                                    'rt_diff': abs(whisky_peak['peak_retention_time'] - sample_peak['peak_retention_time']), 
                                                                    'compounds_whisky': compounds_whisky, 'compounds_sample': compounds_sample, 
                                                                    'len_matches': len([w for w in compounds_whisky for s in compounds_sample if w == s]),
                                                                    'correlation':corr,                                                           
                                                                    }])], ignore_index=True)
    return matchesframe


def filter_peaks(matchesframe, whisky_id, sample_id):
    """
    This function retrieves the peak information from the database and
    uses the splash keys to filter the peaks so that only peaks are
    compared with similar mass spectra.
    :param matchesframe: a dataframe with the information of matching peaks
    :param whisky_id: id of the whisky that the user wants to compare other whiskies with
    :param sample_id: id of the whisky that the user wants to compare to a whisky
    :return: matchesframe = a dataframe with the information of matching peaks
    """
    # Get peaks from database
    peaks_sample1 = requests.get(f"{app_settings.DATABASE_URL}/peaks/?sample_id={sample_id}").json()
    peaks_sample2 = requests.get(f"{app_settings.DATABASE_URL}/peaks/?sample_id={whisky_id}").json()
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
    

def compare_whisky_peaks(whisky_id, sample_list):
    """
    This function acts as the main and calls other functions.
    :param whisky_id: id of the whisky that the user wants to compare other whiskies with
    :param sample_list: a list of whiskies that the user wants to compare to a whisky
    :return: final_df = a dataframe with the percentage of matching peaks between whiskies
    """
    matchesframe = pd.DataFrame()
    for sample_id in sample_list:
        if sample_id != whisky_id:
            matchesframe = filter_peaks(matchesframe, whisky_id, sample_id)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Sort the dataframe by id, retention time and correlation
    sortedframe = matchesframe.sort_values(by=['peak_whisky_id', 'peak_w_rt', 'correlation'])
    
    final_df = pd.DataFrame()
    hit_df = pd.DataFrame()
    for sample_id in sample_list:
        if sample_id != whisky_id:
            matchdf = sortedframe.loc[sortedframe['peak_sample_id'] == sample_id]
            final_df, df = check_peak_order(matchdf.reset_index(), whisky_id, sample_id, final_df)
            hit_df = pd.concat([hit_df, df])
    print(final_df)
    return final_df

if __name__ == "__main__":
    final_df = compare_whisky_peaks(46,[55])
    # final_df = compare_whisky_peaks(2,[1,2,3,4,5,6,7,8,9,10])