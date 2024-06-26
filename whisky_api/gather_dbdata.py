import json
import ijson
import itertools
import glob
import os.path
import pandas as pd
import numpy as np
import pickle
import requests
from natsort import natsorted
from collections import defaultdict
from pyms.BillerBiemann import BillerBiemann, rel_threshold, num_ions_threshold
from pyms.GCMS.IO.ANDI import ANDI_reader
from pyms.IntensityMatrix import build_intensity_matrix
from pyms.Noise.SavitzkyGolay import savitzky_golay
from pyms.Noise.Analysis import window_analyzer
from pyms.Peak.Function import peak_sum_area
from pyms.Peak.List import sele_peaks_by_rt
from pyms.TopHat import tophat
from numpy.linalg import norm
from splash import Spectrum, SpectrumType, Splash
from config import Settings

app_settings = Settings()


def fill_sample_table(sample_data):
    """
    This function fills the sample table in the database.
    :param sample_data: the whisky sample data
    """
    # Insert the data into the data table "sample"
    requests.post(f"{app_settings.DATABASE_IP}/samples/", json=sample_data)


def fill_peak_table(peak_data, code, sample_id=None, aroma_id=None):
    """
    This function fills the peak table in the database.
    :param peak_data: the peak data from a whisky sample or aroma sample.
    :param code: the whisky sample name or aroma id
    :param sample_id: if this variable contains 1, then the peak data is from a whisky
    :param aroma_id: if this variable contains 1, then the peak data is from an aroma
    """
    for peak, data in peak_data.items():
        if sample_id:
            # Get sample from db and get sample id
            sample = requests.get(f"{app_settings.DATABASE_IP}/samples/{code}")
            #data["peak_sample_id"] = int(sample.text.strip("}]").split(",")[-1].split(":")[-1])
            data["peak_sample_id"] = int(json.loads(sample.text)[0]["sample_id"])
        elif aroma_id:
            data["peak_aroma_id"] = int(code)
            
    	# Insert the data into the data table "peak"
        requests.post(f"{app_settings.DATABASE_IP}/peaks/", json=data)


def fill_aroma_table(aroma_data):
    """
    This function fills the aroma table in the database.
    :param aroma_data: the aroma sample data
    """
    requests.post(f"{app_settings.DATABASE_IP}/aromas/", json=aroma_data)
    print(f"{aroma_data['aroma_name']} ingeladen")


def fill_whisky_table(whisky_file):
    """
    This function fills the whisky table in the database.
    :param whisky_file: a file with whisky information
    """
    df = pd.read_excel(whisky_file, skiprows=3, header=None)
    df.columns=['whisky_code', 'whisky_brand', 'whisky_name', 'whisky_age', 'whisky_alcohol_perc',
                'whisky_distillation_freq', 'whisky_distillation_date', 'whisky_bottle_date', 'whisky_filtration',
                'whisky_grain', 'whisky_barrel', 'whisky_distillation_technique', 'whisky_pic_loc']
    df = df.replace({np.nan: None})
    whisky_data = df.to_dict('index')
    for idx, data in whisky_data.items():
        data['whisky_code'] = str(data['whisky_code'])
        requests.post(f"{app_settings.DATABASE_IP}/whisky/", json=data)
        print(data)


def tic_cleaner(data):
    """
    This function cleans the TIC data and creates a cleaned up TIC file.
    :param data: a cdf file
    :return: tic = a cleaned up version of the tic of the data.
    :return: im = a cleaned up version of the intensity matrix of the data.
    """    
    # Retrieve the tic data, number of scans and number of masses from the data and build an intensity matrix.
    tic = data.tic
    im = build_intensity_matrix(data)
    n_scan, n_mz = im.size
    # For every mass, smooth the graph using the savitzky_galoy smoother using standard parameters.
    # https://aip.scitation.org/doi/pdf/10.1063/1.4822961
    # next, it removes the baseline using the tophat algorithm.
    # https://arxiv.org/pdf/1603.07082.pdf
    # both algorithms come from pymassspec.
    # the cleaned up mass graph is then inserted back into the intensity matrix.
    # once done, the intensity matrix is used to create a cleaned up TIC file.
    for ii in range(n_mz):
        ic = im.get_ic_at_index(ii)
        smoothed = savitzky_golay(ic)
        ic = tophat(smoothed, struct="1.5m")
        im.set_ic_at_index(ii, ic)
    tic.intensity_array = np.sum(np.array(im.intensity_array), axis=1)
    return tic, im


def get_peak_compound(compound_dict, s_intensities, s_masses, s_splash2, s_splash3):
    """
    This function matches compounds to a peak. The correlation is 
    calculated between the peak MS and the compound MS. These compounds 
    are added to a dataframe, sorted and the top 10 compounds get returned. 
    :param compound_dict: dictionary with compound data
    :param s_intensities: list with intensities from the peak
    :param s_masses: list with masses from the peak
    :param s_splash2: splash key component 2 (in base-3) from the peak
    :param s_splash3: splash key component 3 from the peak
    :return: list with the names of the top 10 matching compound
    """
    compound_list = []
    # Loop through dict of compounds
    for key, value in compound_dict.items():
        # Change the splash key component 2 from base-36 to base-3
        # https://www.unitconverters.net/numbers/base-36-to-base-3.htm
        c_splash2 = np.base_repr(int(str(str(key.split("-")[1])),36),base=3).rjust(10,"0")
        # Calculate the difference between the splash key components
        diff_splash2 = sum([abs(int(a)-int(b)) for a,b in zip(s_splash2, c_splash2)])
        diff_splash3 = sum([abs(int(a)-int(b)) for a,b in zip(str(s_splash3),str(key.split("-")[2]))])
        # If the difference between components is less than or equal to 4, calculate the
        # correlation between the peak MS and the compound MS. If correlation is greater than
        # or equal to 0.7, the compound is added to a list.
        if diff_splash2 <= 4 and diff_splash3 <= 4:
            hit=0
            peak_matches=[]
            query_matches=[]
            c_masses = json.loads(value['masses'])
            c_intensities = json.loads(value['intensities'])
            for ms, mc in itertools.product(s_masses,c_masses):
                if abs(mc-ms) <= 0.5:
                    hit+=1
                    peak_matches.append(s_intensities[s_masses.index(ms)])
                    query_matches.append(c_intensities[c_masses.index(mc)])
            if hit >= 1:
                product = sum([x*y for x,y in zip(peak_matches, query_matches)])
                corr = np.power(product / (norm(s_intensities)*norm(c_intensities)),2)
            if corr >= 0.7:
                compound_list.append([value['compound'], corr])

    unique_compound_list = []
    # Convert the list with matching compounds to a dataframe
    df = pd.DataFrame(compound_list, columns=['name', 'corr'])
    # Remove the multiple occurring compounds from the df and convert back to a list.
    for compound in list(set([x[0] for x in compound_list])):
        unique_compound_list.append(df.loc[df['name'] == compound].sort_values(by=['corr'], ascending=False).head(1).values.flatten().tolist())
    # Sort the list of compounds by greatest correlation
    unique_compound_list.sort(key = lambda x: x[1], reverse=True)
    return [x[0] for x in unique_compound_list[0:10]]


def peak_finder(tic, im, factor, compound_dict):
    """
    This function detects peaks and retrieves all relevant peak information.
    :param tic: tic data
    :param im: intensity matrix
    :param factor: the factor by which the noise is to be multiplied
    :param compound_dict: a dataframe with compound information
    :return: profile = a dataframe with all peak information
    """
    # Determine the signal noise using the window_analyzer from pymassspec with standard settings.
    noise = window_analyzer(tic, rand_seed=50)
    # Use BillerBiemann from pymassspec to get a list of peaks.
    peak_list = BillerBiemann(im, points=50, scans=15) #25,5
    # First, for each peak, we filter out any mass channel which value is under 2% of the max mass channel value.
    # if a peak no longer has any apexes as a result. it is filtered out.
        # Remove ions with relative intensities less than the given relative percentage of the maximum intensity.
        # Parameters:
        # pl (Sequence[Peak])
        # percent (float) – Threshold for relative percentage of intensity. Default 2%.
        # copy_peaks (bool) – Whether the returned peak list should contain copies of the peaks. Default True.
    pl = rel_threshold(peak_list, percent=4) #percent=3

    # next, we filter out any peak which does not have 3 mass channels who's value go above noise * factor
    # This step is where most false positives are removed.
        # Remove Peaks where there are fewer than n ions with intensities above the given threshold.
        # Parameters:
        # pl (Sequence[Peak])
        # n (int) – Minimum number of ions that must have intensities above the cutoff.
        # cutoff (float) – The minimum intensity threshold.
        # copy_peaks (bool) – Whether the returned peak list should contain copies of the peaks. Default True.
    pl = num_ions_threshold(pl, n=2, cutoff=noise*factor) #n=4, cutoff=noise*50

    # Select peaks from a retention time range.
    pl = sele_peaks_by_rt(pl,("0s","1830s"))

    # create a nested dictionary to hold the peak data.
    profile = defaultdict(dict)
    for p in pl:
        peak_id = p.bounds[1]
        # get the area, and create a list to hold the ms profile, masses and intensities
        area = peak_sum_area(im, p)
        p.area = area
        p_ms = []
        masses = []
        intensities = []
        # get the mass spectrum and the value of the highest mass of said spectrum
        ms = im.get_ms_at_index(peak_id)
        mx = max(ms.intensity_list)
        # Recreate the intensities in the MS value based on the maximum intensity value.
        # The maximum intensity will now be 100 while every other intensity is calculated by:
        # intensityvalue / max_value * 100.
        for x in range(len(ms.mass_list)):
            v = ms.intensity_list[x] / mx * 100
            #ms values which are too low compared to the best m/z value are deemed noise and discarded
            if v >= 0.1:
                masses.append(round(ms.mass_list[x], 2))
                intensities.append(round(ms.intensity_list[x], 2))
                p_ms.append((ms.mass_list[x], v))
        # generate a splash code for the ms spectrum, json dump the masses and intensities
        splash = Splash().splash(Spectrum(p_ms, SpectrumType.MS))
        compound_list = get_peak_compound(compound_dict, intensities, masses, np.base_repr(int(str(splash.split("-")[1]),36),base=3).rjust(10,"0"), splash.split("-")[2])
        masses = json.dumps(masses)
        intensities = json.dumps(intensities)
        height = round(tic.get_intensity_at_index(peak_id))
        if compound_list:  
            profile[p.rt] = {"peak_splash": splash, "peak_retention_time": float(p.rt), "peak_masses": str(masses),
                            "peak_intensities": str(intensities),"peak_height": height, "peak_area": round(p.area), "peak_compound": json.dumps(compound_list)}
        else:
            profile[p.rt] = {"peak_splash": splash, "peak_retention_time": float(p.rt), "peak_masses": str(masses),
                            "peak_intensities": str(intensities),"peak_height": height, "peak_area": round(p.area), "peak_compound": json.dumps(["N/A"])}
    print(len(profile))
    return profile
   
    
def gather_sampledata(cdf_file, tic_loc, compound_dict):
    """
    This function reads the whisky cdf file and makes a TIC file. It
    retrieves the whisky name, file locations and the peak data to fill 
    the database tables sample and peak. 
    :param cdf_file: location of the whisky cdf file
    :param tic_loc: location where the whisky tic file should be stored
    :param compound_dict: a dataframe with compound information
    """
    # Get file name
    name = cdf_file.split("/")[-1].split(".")[0].strip("\n")
    # Gather the file type.
    filetype = cdf_file.split(".")[-1].lower()
    # Check file type.
    if filetype == "cdf":
        data = ANDI_reader(cdf_file)
        # Clean the tic (total ion chromatogram) which can then be used to fill the database, the tic can
        # then be used to search for peaks.
        tic, im = tic_cleaner(data)
        tic_file = os.path.join(tic_loc, name)
        pickle.dump(tic, open(tic_file, "wb"))
        # save data in dictionarys
        sample_data = {"sample_code": name, "sample_file_location": cdf_file, "sample_tic_location": tic_file}
        peak_data = peak_finder(tic, im, 60, compound_dict)
        fill_sample_table(sample_data)
        fill_peak_table(peak_data, name, sample_id=1)
    else:
        print("Data type not supported.\nSupported types are: .cdf")     
    

def gather_aromadata(cdf_file, tic_loc, aroma_file, compound_dict):
    """
    This function reads the aroma cdf files and makes TIC files. It
    retrieves the aroma data and the peak data to fill the database tables
    le_nez_du_whisky_aroma and peak. 
    :param cdf_file: location of all the aroma cdf files
    :param tic_loc: location where all the aroma tic files should be stored
    :param aroma_file: a file with information about every aroma
    :param compound_dict: a dataframe with compound information
    """
    # Open and read file
    aromas = open(aroma_file, "r").readlines()
    aroma_files = natsorted(glob.glob(cdf_file +"/*"))
    print(aroma_files)
    for aroma in aromas[1:]:
        aroma = aroma.strip("\n").split(",,")
        file_name = aroma_files[int(aroma[0])-1]
        print(file_name)
        data = ANDI_reader(file_name)
        # Clean the tic (total ion chromatogram) which can then be used to fill the database, the tic can
        # then be used to search for peaks.
        tic, im = tic_cleaner(data)
        tic_file = os.path.join(tic_loc, str("_".join(aroma[1].split(" "))))
        print(tic_file)
        pickle.dump(tic, open(tic_file, "wb"))
        aroma_info = {"aroma_id": aroma[0],"aroma_name": aroma[1], "aroma_notes": aroma[2], "aroma_annotation": aroma[3], "aroma_description": aroma[4],
                    "aroma_file_location": file_name, "aroma_tic_location": tic_file}
        peak_data = peak_finder(tic, im, 175, compound_dict)
        fill_aroma_table(aroma_info)
        fill_peak_table(peak_data, aroma_info["aroma_id"], aroma_id=1)


def get_compound_from_json(MONA_file):
    """
    This function retrieves the name, splash key, masses and intensities
    from all the compounds in the json file. This file contains all
    GC-MS compounds from the MoNA database. This file can be downloaded
    from https://mona.fiehnlab.ucdavis.edu/downloads. 
    :param MONA_file: a file containing all GC-MS compounds from the MoNA database.
    :return: compound_dict = a dataframe with compound information
    """
    # Create a dictionary to store the useful parts of the MoNA database
    compound_dict = defaultdict(dict)
    # Open and parse the database
    d = open(MONA_file,'rb')
    for id in ijson.items(d, 'item'):
        # Retrieve the splash code
        splash = id['splash']['splash'] 
        # Retrieve the compound name. There are many names in the database but this retrieves the common name.
        name = id['compound'][0]['names'][0]['name'].lower()
        # Retrieve the mass spectrum
        sp = [float(ele) for combi in id['spectrum'].split(" ") for ele in combi.split(":")]
        # Seperate the masses and the intensities
        masses = json.dumps(sp[0::2])
        intensities = json.dumps(sp[1::2])
        # If the MS profile was made by a GC-MS machine and the splash code is not yet in the dictionary, add it.
        # The reason we don't add duplicate splash codes is because only profile's with an identical MS profile
        # have identical splash codes.
        if id['tags'][-1]['text'] == "GC-MS" and splash not in compound_dict.keys():
            compound_dict[splash] = {"compound": name, "masses": masses, "intensities": intensities}
    return compound_dict 


def main(input, data_type):
    """
    The main checks if the given data is a whisky sample,
    aroma sample or whisky information.
    """
    if data_type == "sample":
        compound_dict = get_compound_from_json(input[2])
        new_path = str(input[0])+"/*"
        whisky_samples = glob.glob(new_path)
        for sample in whisky_samples:
            gather_sampledata(sample, input[1], compound_dict)
    elif data_type == "aroma":
        compound_dict = get_compound_from_json(input[3])
        gather_aromadata(input[0], input[1], input[2], compound_dict)
    elif data_type == "whisky":
        fill_whisky_table(input)
    else:
        print("Nothing")


if __name__ == "__main__":
    main(snakemake.params.locations, snakemake.params.data_type)
