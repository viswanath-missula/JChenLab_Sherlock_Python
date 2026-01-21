# Import dependencies
import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pingouin as pg
import nibabel as nib

# Define helper functions
def compute_lagged_corr(x, y, max_lag):
    # Computes lagged correlation between arrays x and y
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]
        corrs.append(corr)
    
    return np.array(corrs)

def corr_inter(A, B):
    # Computes and returns a matrix of the pairwise correlation coefficient between each pair of columns in the input matrices A and B
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    cov = A.T @ B
    stdA = np.sqrt((A**2).sum(axis=0))
    stdB = np.sqrt((B**2).sum(axis=0))
    return cov / np.outer(stdA, stdB)

def corr_match_columns(A, B):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    num = np.sum(A * B, axis=0)
    denom = np.sqrt(np.sum(A**2, axis=0)) * np.sqrt(np.sum(B**2, axis=0))
    return num / denom

def return_paths(roi, subject):
    movie_path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\subjects\s" + str(subject) + r"\sherlock_movie" + roi + r"_sherlock_movie_s" + str(subject) + r".mat"
    recall_path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\subjects\s" + str(subject) + r"\sherlock_recall" + roi + r"_sherlock_recall_s" + str(subject) + r".mat"
    return (movie_path, recall_path)

def return_func_data_path(subject):
    out = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\subjects\s" + str(subject) + r"\sherlock_movie\sherlock_movie_s" + str(subject) + r".mat"
    return out


def test_shape(lst):
    print(lst)
    print()
    for x in lst:
        try:
            print(x.shape)
        except:
            print("list elements are not numpy arrays")

def extract_fMRI_data(roi, numsubs):
    rdata_m = []
    rdata_r = []

    for x in range(1, numsubs+1):
        (mv_path, rcl_path) = return_paths(roi, x)
        mv_data = loadmat(mv_path)
        rcl_data = loadmat(rcl_path)
        rdata_m.append(mv_data["rdata"])
        rdata_r.append(rcl_data["rdata"])
    
    return (rdata_m, rdata_r)

def produce_ev_encoding(ev):
    subj_data = ev[list(ev.dtype.names)[0]][0, 0]
    fullSL1 = subj_data["fullSL1"]
    fullSL2 = subj_data["fullSL2"] + fullSL1[-1, -1]
    ev_encoding = np.concatenate((fullSL1, fullSL2))
    return ev_encoding

def compute_mean_voxel_activity(rdata_movie_ls, numsubs, num_TR):
    # Define temporary arrays
    roitc_temp = []
  
    # Compute mean voxel activity at each TR for each subject
    for subject_number in range(numsubs):
        rdata = rdata_movie_ls[subject_number]
        sum_rdata = np.sum(rdata, axis=0)/num_TR
        roitc_temp.append(sum_rdata)

    # Organize data
    roitc_untransposed = np.array(roitc_temp)
    roitc = np.matrix.transpose(np.array(roitc_temp))

    # Compute mean voxel activity at each TR across all subjects
    roitc_mean = np.sum(roitc, axis=1)/(numsubs)
    return roitc_mean, roitc_untransposed

def compute_dataset_lagged_corrs(roitc_mean, roitc_untransposed, numsubs, set_lag=50):
    lags = np.arange(-set_lag, set_lag + 1)
    lag_corr_ls = []

    for curr_sub in range(numsubs):
        curr_roitc = roitc_untransposed[curr_sub]
        mean_roitc_curr_excl = (roitc_mean*numsubs - curr_roitc)/(numsubs-1)
        lag_corr = compute_lagged_corr(curr_roitc, mean_roitc_curr_excl, set_lag)
        lag_corr_ls.append(lag_corr)

    return lag_corr_ls, lags

def compute_corr_mat(rdata_movie_ls, numsubs):
    corr_mat_ls = []
    rdata_sum = sum(rdata_movie_ls)

    for curr_sub in range(numsubs):
        curr_data = rdata_movie_ls[curr_sub]
        other_subj_data = (rdata_sum-curr_data)/(numsubs-1)
        corr_mat = corr_inter(other_subj_data, curr_data)
        corr_mat_ls.append(corr_mat)

    avg_corr_mat = sum(corr_mat_ls)/numsubs
    avg_corr_mat_diag = np.diag(avg_corr_mat)
    return avg_corr_mat, avg_corr_mat_diag

def return_ticks(end_of_range, spacing):
    ticks = [x for x in range(0, end_of_range, spacing)]
    tick_labels = [str(x) for x in ticks]
    return ticks, tick_labels

def compute_recall_behavior(ev, subj_chosen_indexed_at_one):
    subj_chosen = subj_chosen_indexed_at_one - 1
    subj_ID = list(ev.dtype.names)[subj_chosen]
    subj_data = ev[subj_ID][0, 0]

    # Extract movie and recall related data
    fullSL1 = subj_data["fullSL1"]
    fullSL2 = subj_data["fullSL2"] + fullSL1[-1, -1]
    all_movie_scenetimes = np.concatenate((fullSL1, fullSL2))
    full_movie_length = all_movie_scenetimes[-1][-1]

    recall_scenetimes = subj_data["events"]["freerecall"][-1, -1]
    recall_length = recall_scenetimes[-1, -1]
    movie_SL1 = subj_data["events"]["SL1"][0, 0]
    movie_SL2 = subj_data["events"]["SL2"][0, 0]+fullSL1[-1, -1]
    subj_movie_scenetimes = np.concatenate((movie_SL1, movie_SL2))

    # Compute and plot the display matrix
    recall_behavior_mat = np.zeros((int(full_movie_length), int(recall_length)))

    for i in range(subj_movie_scenetimes.shape[0]):
        movie_scene_start = subj_movie_scenetimes[i,0]
        movie_scene_end = subj_movie_scenetimes[i,1]
        recall_scene_start = recall_scenetimes[i,0]
        recall_scene_end = recall_scenetimes[i,1]
        recall_behavior_mat[movie_scene_start:movie_scene_end+1, recall_scene_start:recall_scene_end+1] = 1
    
    return recall_behavior_mat

def compute_voxel_pattern_averages_movrec(ev, numsubs, roi_chosen):
    names = ["s"+str(x+1) for x in range(numsubs)]
    all_subs_iM_events = []
    all_subs_iFr_events = []
    all_subs_50M_events = []
    all_subs_50Fr_events = []

    for n in range(len(names)):
        # Load all relevant data
        ev_prsnt = ev[names[n]][0, 0]
        movie_SL1 = ev_prsnt["events"]["SL1"][0, 0]
        movie_SL2 = ev_prsnt["events"]["SL2"][0, 0] + ev_prsnt["fullSL1"][-1, -1]
        ev_encoding = np.concatenate((movie_SL1, movie_SL2))

        (m_path, fr_path) = return_paths(roi_chosen, n+1)
        m_data = loadmat(m_path)["rdata"]
        fr_data = loadmat(fr_path)["rdata"]

        # First handle iM avg scenes for sherlock_movie
        m_events = np.zeros((m_data.shape[0], ev_encoding.shape[0]))

        for e in range(ev_encoding.shape[0]):
            m_events[:, e] = np.mean(m_data[:, ev_encoding[e, 0]:ev_encoding[e, 1]+1], axis=1)

        all_subs_iM_events.append(m_events)
        # Now handle iFr avg scenes for sherlock_recall
        freerecall = ev_prsnt["events"]["freerecall"][0, 0]
        fr_events = np.zeros((fr_data.shape[0], freerecall.shape[0]))

        for e in range(freerecall.shape[0]):
            fr_events[:, e] = np.mean(fr_data[:, freerecall[e, 0]:freerecall[e, 1]+1], axis=1)

        all_subs_iFr_events.append(fr_events)

        # Now handle 50M avg scenes
        fullSL1 = ev_prsnt["fullSL1"]
        fullSL2 = ev_prsnt["fullSL2"] + fullSL1[-1, -1]
        ev_encoding = np.concatenate((fullSL1, fullSL2))
        m_events = np.zeros((m_data.shape[0], ev_encoding.shape[0]))

        for e in range(ev_encoding.shape[0]):
            m_events[:, e] = np.mean(m_data[:, ev_encoding[e, 0]:ev_encoding[e, 1]+1], axis=1)

        all_subs_50M_events.append(m_events)

        # Now handle 50Fr avg scenes
        num_Events = ev_prsnt["SL_event_num"][-1, -1]
        fr_num = ev_prsnt["events"]["freerecallnum"][0, 0]
        fr_events = np.zeros((fr_data.shape[0], freerecall.shape[0]))
        temp = np.full((fr_data.shape[0], num_Events), np.nan)
        tempcount = np.zeros((1, num_Events))

        for e in range(freerecall.shape[0]):
            fr_events[:, e] = np.mean(fr_data[:, freerecall[e, 0]:freerecall[e, 1]+1], axis=1)

        for i in range(len(fr_num)):
            if np.isnan(fr_num[i]):
                continue
            fr_num_idx = int(fr_num[i]-1)
            if np.isnan(temp[0, fr_num_idx]):
                temp[:, fr_num_idx] = fr_events[:, i]
            else:
                temp[:, fr_num_idx] = np.nansum(np.vstack([temp[:, fr_num_idx], fr_events[:, i]]), axis=0)
            tempcount[0, fr_num_idx] += 1
            
        all_subs_50Fr_events.append(temp / tempcount[np.newaxis, :])

    all_subs_50M_events = np.transpose(np.array(all_subs_50M_events), (1, 2, 0))
    all_subs_50Fr_events = np.transpose(np.squeeze(np.array(all_subs_50Fr_events), axis=1), (1, 2, 0))

    return (all_subs_iM_events, all_subs_iFr_events, all_subs_50M_events, all_subs_50Fr_events)

def compute_scene_level_pattern_corr(all_subs_iM_events, all_subs_iFr_events, all_subs_50M_events, all_subs_50Fr_events, numsubs):
    sdiag_movrec_withinsubj = []
    mdiag_movrec_withinsubj = []
    movrec_withinsubj_corrmat = []
    movmov_btwnsubj_corrmat = []
    sdiag_movmov_btwnsubj = []
    mdiag_movmov_btwnsubj = []
    recrec_btwnsubj_corrmat = []
    sdiag_recrec_btwnsubj = []
    mdiag_recrec_btwnsubj = []

    for n in range(numsubs):
        sdiag_movrec_withinsubj.append(np.diagonal(corr_inter(all_subs_iM_events[n], all_subs_iFr_events[n])))
        mdiag_movrec_withinsubj.append(np.nanmean(sdiag_movrec_withinsubj[n]))

        movrec_withinsubj_corrmat.append(corr_inter(all_subs_50Fr_events[:,:,n], all_subs_50M_events[:,:,n]))

        others = [i for i in range(numsubs) if i != n]
        movmov_btwnsubj_corrmat.append(corr_inter(all_subs_50M_events[:,:,n], np.nanmean(all_subs_50M_events[:,:,others], axis=2)))
        sdiag_movmov_btwnsubj.append(np.diagonal(corr_inter(all_subs_50M_events[:,:,n], np.nanmean(all_subs_50M_events[:,:,others], axis=2))))
        mdiag_movmov_btwnsubj.append(np.nanmean(sdiag_movmov_btwnsubj[n]))

        recrec_btwnsubj_corrmat.append(corr_inter(all_subs_50Fr_events[:,:,n], np.nanmean(all_subs_50Fr_events[:,:,others], axis=2)))
        sdiag_recrec_btwnsubj.append(np.diagonal(corr_inter(all_subs_50Fr_events[:,:,n], np.nanmean(all_subs_50Fr_events[:,:,others], axis=2))))
        mdiag_recrec_btwnsubj.append(np.nanmean(sdiag_recrec_btwnsubj[n]))

    movrec_withinsubj_corrmat = np.transpose(np.array(movrec_withinsubj_corrmat), (1, 2, 0))
    movmov_btwnsubj_corrmat = np.transpose(np.array(movmov_btwnsubj_corrmat), (1, 2, 0))
    recrec_btwnsubj_corrmat = np.transpose(np.array(recrec_btwnsubj_corrmat), (1, 2, 0))

    return (movrec_withinsubj_corrmat, movmov_btwnsubj_corrmat, recrec_btwnsubj_corrmat, mdiag_movmov_btwnsubj, mdiag_movrec_withinsubj, mdiag_recrec_btwnsubj)

def get_predictors_1000segs(content_label, dataframe, plotme=False):
    content_var = []
    scene_titles = []
    scene_start_seg = []

    for row in range(dataframe.shape[0]):
        if type(dataframe.iloc[row,5]) == type(""):
            if dataframe.iloc[row,5].strip() != "":
                scene_titles.append(dataframe.iloc[row,5].strip())
                scene_start_seg.append(row)

    if content_label == "indoor":
        vec = dataframe.iloc[:,7]
        content_var = [1 if x == "Indoor" else 0 for x in vec]

    elif content_label == "outdoor":
        vec = dataframe.iloc[:,7]
        content_var = [1 if x == "Outdoor" else 0 for x in vec]

    elif content_label == "numpersons":
        vec = dataframe.iloc[:,8]
        for row in vec:
            charlist = set([x.strip() for x in row.split(",")])
            count = len(charlist)-1 if "Nobody" in charlist else len(charlist)
            content_var.append(count)

    elif content_label == "music":
        vec = dataframe.iloc[:,13]
        content_var = [1 if x == "Yes" else 0 for x in vec]

    elif content_label == "speaking":
        vec = dataframe.iloc[:,10]
        content_var = [1 if type(x) == type("") else 0 for x in vec]
    
    elif content_label == "writwords":
        vec = dataframe.iloc[:,14]
        content_var = [1 if type(x) == type("") else 0 for x in vec]

    elif len(content_label.split("_")) == 2 and content_label.split("_")[1] == "onscreen":
        charname = content_label.split("_")[0]
        vec = dataframe.iloc[:,8]
        for row in vec:
            charlist = set([x.strip() for x in row.split(",")])
            content_var.append(1 if charname in charlist else 0)
    
    elif len(content_label.split("_")) == 2 and content_label.split("_")[1] == "speaking":
        charname = content_label.split("_")[0]
        vec = dataframe.iloc[:,10]
        for row in vec:
            charlist = set([x.strip() for x in row.split(",")])
            content_var.append(1 if charname in charlist else 0)

    elif content_label == "locations":
        vec = dataframe.iloc[:,11]
        content_var = np.zeros((vec.shape[0]))
        unique = list(set(vec))
        for count_id in range(len(unique)):
            mask = vec.isin([unique[count_id]])
            content_var[mask] = count_id+1
    
    elif content_label == "arousal":
        content_var = (dataframe.iloc[:,15]+dataframe.iloc[:,17]+dataframe.iloc[:,19]+dataframe.iloc[:,21])/4.0

    elif content_label == "valence":
        content_var = np.zeros((dataframe.shape[0]))
        for x in [16,18,20,22]:
            mask_plus = dataframe.iloc[:,x].isin(["+"])
            mask_minus = dataframe.iloc[:,x].isin(["-"])
            content_var[mask_plus] += 1
            content_var[mask_minus] -= 1
        content_var /= 4.0

    if plotme:
        plt.figure(figsize=(15, 15))
        plt.gcf().set_facecolor('white')
        plt.plot(content_var, color='r')
        plt.title(content_label.replace('_', '-'), fontsize=18)
        plt.xlabel('Time')

        maxm = np.max(content_var)
        minm = np.min(content_var)
        plt.ylim([minm - 1.5, maxm + 0.5])
        plt.yticks(np.arange(0, max(content_var) + 1, 1))

        for ss in range(len(scene_start_seg)):
            plt.plot([scene_start_seg[ss], scene_start_seg[ss]],
                    [minm - 2, maxm + 0.5],
                    linestyle='--', color="gray")
            y_offset = (ss % 10) * -0.1 - 0.2 + minm
            plt.text(scene_start_seg[ss], y_offset, scene_titles[ss], fontsize=8, backgroundcolor="white")

        plt.plot(content_var, color='r')

    return (np.array(content_var).T, content_label, scene_titles)

def get_predictors_from_labelnames(labels_df, labelnames, name_short_len=5):
    labeltc = np.zeros((labels_df.shape[0], len(labelnames)))
    labeltc_TRs_rep = []
    rowTR_rep = []
    SL_short_segments = labels_df.iloc[:,3:5].iloc

    for label_num in range(len(labelnames)):
        labeltc[:,label_num] = get_predictors_1000segs(labelnames[label_num], labels_df)[0]

    for k in range(labeltc.shape[0]):
        rowvals = labeltc[k, :]
        nTRs_this_seg = SL_short_segments[k, 1] - SL_short_segments[k, 0] + 1
        segblock = np.tile(rowvals, (int(nTRs_this_seg), 1))
        rowTR_rep += list(np.arange(SL_short_segments[k, 0], SL_short_segments[k, 1] + 1).reshape(1, -1)[0])
        labeltc_TRs_rep.append(segblock)

    labeltc_TRs_rep = np.vstack(labeltc_TRs_rep)
    rowTR_rep = np.array(rowTR_rep)
    labeltc_TRs = np.zeros((int(np.max(rowTR_rep)), labeltc_TRs_rep.shape[1]))

    for j in range(labeltc_TRs.shape[0]):
        ii = [True if rowTR_rep[n] == j else False for n in range(len(rowTR_rep))]
        newrow = np.mean(labeltc_TRs_rep[ii,:], 0)
        labeltc_TRs[j,:] = newrow

    labeltc_TRs = labeltc_TRs[1:,:]

    for n in range(labeltc_TRs.shape[1]):
        labeltc_TRs[:,n] = (labeltc_TRs[:,n]-np.min(labeltc_TRs[:,n]))/(np.max(labeltc_TRs[:,n])-np.min(labeltc_TRs[:,n]))

    labelnames_short = [x[:name_short_len].replace("_","") for x in labelnames]
    return (labeltc_TRs, labelnames_short)

# Define dataset parameters (number of subjects, roi, etc...) and extract commonly used data
#roi_chosen = r"\aud_early"
roi_chosen = r"\pmc_nn"
num_subj = 17
path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata"
ev_path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\subjects\sherlock_allsubs_events.mat"
ev = loadmat(ev_path)["ev"][0, 0]
custom_palette = matplotlib.colors.LinearSegmentedColormap.from_list("my_blend", ["black", "red", "orange", "yellow", "white"])
colorlist = sns.color_palette("husl", num_subj)
labels = ["s"+str(x+1) for x in range(num_subj)] + ["mean"]

# Extract fMRI data for each subject
rdata_movie_ls, rdata_recall_ls = extract_fMRI_data(roi_chosen, num_subj)

# Extract number of time points
num_TR = len(rdata_movie_ls[0][0])

# Compute mean voxel activity at each TR for each subject
roitc_mean, roitc_untransposed = compute_mean_voxel_activity(rdata_movie_ls, num_subj, num_TR)

# Plot mean voxel activity per subject + across all subjects
for iter in range(num_subj):
    plt.plot(roitc_untransposed[iter], color=colorlist[iter])

plt.plot(roitc_mean, color="black")
plt.xlabel("TR")
plt.ylabel("Z")
plt.legend(labels)
plt.show()
plt.clf()
plt.close('all')

# Compute lagged correlation between each subject and average of other subjects
lag_corr_ls, lags = compute_dataset_lagged_corrs(roitc_mean, roitc_untransposed, num_subj, 50)

# Plot lagged correlations
for iter in range(num_subj):
    plt.plot(lag_corr_ls[iter], color=colorlist[iter])

plt.xlabel("TR")
plt.ylabel("Z")
plt.legend(labels[:-1])
plt.show()
plt.clf()
plt.close('all')

# Compute and plot timepoint-by-timepoint cross-subject pattern correlation matrices for movie-movie
avg_corr_mat, avg_corr_mat_diag = compute_corr_mat(rdata_movie_ls, num_subj)

ticks, tick_labels = return_ticks(num_TR, 200)

avg_corr_mat_heatmap = sns.heatmap(avg_corr_mat, vmin=-0.3, vmax=0.3, cmap=custom_palette)
avg_corr_mat_heatmap.set(xlabel="Time (TR)", ylabel="Time (TR)",
                         xticks=ticks, xticklabels=tick_labels,
                         yticks=ticks, yticklabels=tick_labels)
plt.show()
plt.clf()
plt.close('all')

# Plot recall behavior for a specific subject
subj_chosen = 5
recall_behavior_mat = compute_recall_behavior(ev, subj_chosen)

plt.imshow(recall_behavior_mat, aspect="auto", cmap='Greys_r')
plt.title("Subject " + str(subj_chosen))
plt.xlabel("Recall time (TRs)")
plt.ylabel("Movie time (TRs)")
plt.show()
plt.clf()
plt.close('all')

# Compute matrices to find voxel pattern averages for each scene
roi_chosen = r"\pmc_nn"
all_subs_iM_events, all_subs_iFr_events, all_subs_50M_events, all_subs_50Fr_events = compute_voxel_pattern_averages_movrec(ev, num_subj, roi_chosen)

# Start computing scene-level pattern correlations
(movrec_withinsubj_corrmat, movmov_btwnsubj_corrmat, recrec_btwnsubj_corrmat,
 mdiag_movmov_btwnsubj, mdiag_movrec_withinsubj, mdiag_recrec_btwnsubj) = compute_scene_level_pattern_corr(all_subs_iM_events, all_subs_iFr_events, all_subs_50M_events, all_subs_50Fr_events, num_subj)

# Plot the relevant graphs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ticks, tick_labels = return_ticks(movrec_withinsubj_corrmat.shape[0], int(movrec_withinsubj_corrmat.shape[0]/10))

sns.heatmap(np.nanmean(movrec_withinsubj_corrmat, axis=2),
            vmin=-0.5, vmax=0.5, cmap=custom_palette,
            ax=axes[0], square=True)
axes[0].set(title="Movie-Recall Within Subject",
            xticks=ticks, xticklabels=tick_labels,
            yticks=ticks, yticklabels=tick_labels)

sns.heatmap(np.nanmean(movmov_btwnsubj_corrmat, axis=2),
            vmin=-0.5, vmax=0.5, cmap=custom_palette,
            ax=axes[1], square=True)
axes[1].set(title="Movie-Movie Between Subjects",
            xticks=ticks, xticklabels=tick_labels,
            yticks=ticks, yticklabels=tick_labels)

sns.heatmap(np.nanmean(recrec_btwnsubj_corrmat, axis=2),
            vmin=-0.5, vmax=0.5, cmap=custom_palette,
            ax=axes[2], square=True)
axes[2].set(title="Recall-Recall Between Subjects",
            xticks=ticks, xticklabels=tick_labels,
            yticks=ticks, yticklabels=tick_labels)

plt.show()
plt.clf()
plt.close('all')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(mdiag_movrec_withinsubj, ax=axes[0])
axes[0].set(title="Spatial Corr Within Subject Movie vs. Recall", xlabel="Subject", ylabel="R")

sns.barplot(mdiag_movmov_btwnsubj, ax=axes[1])
axes[1].set(title="Spatial Corr Between Subjects Movie vs. Movie", xlabel="Subject", ylabel="R")

sns.barplot(mdiag_recrec_btwnsubj, ax=axes[2])
axes[2].set(title="Spatial Corr Between Subjects Recall vs. Recall", xlabel="Subject", ylabel="R")

plt.show()
plt.clf()
plt.close('all')

# Read and plot movie feature labels
labels_path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\subjects\Sherlock_Segments_1000_NN_2017.xlsx"
labels_df = pd.read_excel(labels_path).drop([480,481])

# Plot just one label
(content_var, content_label, scene_titles) = get_predictors_1000segs("Sherlock_onscreen", labels_df, plotme=True)
#plt.show()
plt.clf()
plt.close('all')

# Get predictors and define plotting components
labelnames = ['speaking','numpersons','locations','arousal','valence','indoor', 'music','writwords','Sherlock_onscreen', 'John_onscreen']
labeltc_TRs, labelnames_short = get_predictors_from_labelnames(labels_df, labelnames, name_short_len=5)
scenestart = produce_ev_encoding(ev)[:,0]
step = labeltc_TRs.shape[0]/51
tvec = np.arange(step, labeltc_TRs.shape[0], step)[:50]
xval = 12.5
labeltc_TRs_padded = np.hstack((np.full((labeltc_TRs.shape[0], 1), np.nan), labeltc_TRs))
labelnames_short = [""] + labelnames_short

# Plot the result
plt.figure(figsize=(15, 15))
plt.gcf().set_facecolor('white')
plt.imshow(labeltc_TRs_padded, aspect='auto', cmap='hot', origin='upper', interpolation='none')
plt.xlim([-0.2, 25])
plt.xticks(ticks=np.arange(11), labels=labelnames_short, fontsize=8)
plt.tick_params(labelsize=9)
plt.plot([10.8]*len(scenestart), scenestart, 'b<', linewidth=3)
plt.plot([0.2]*len(scenestart), scenestart, 'b>', linewidth=3)

for t in range(len(tvec)):
    plt.text(xval, tvec[t], scene_titles[t], fontsize=10)
    plt.plot([10.8, xval], [scenestart[t], tvec[t]], 'b-')

plt.show()
plt.clf()
plt.close('all')

# Regress timecourse onto brain map
def extract_roicorr(roi):
    path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\intersubj\roicorr\ "[:-1] + roi + "_sherlock_movie_roicorr.mat"
    roicorr = loadmat(path)
    return roicorr


roi_chosen = "bilateral_hipp"
hipp_summary_file = extract_roicorr(roi_chosen)
hipptc = hipp_summary_file["meantc"]

s1 = loadmat(return_func_data_path(1))
rmap = corr_match_columns(s1["data"].conj().T, hipptc)
rmap_3d = rmap.reshape(tuple(s1["datasize"][:, :3].flatten()))
print(rmap_3d.shape)

mni_path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\standard\MNI152_T1_3mm_brain.nii"
mni_brain = nib.load(mni_path)
hdr_copy = mni_brain.header
hdr_copy["cal_max"] = 0.4
hdr_copy["cal_min"] = 0.1
new_mni = nib.Nifti1Image(rmap_3d, affine=mni_brain.affine, header=hdr_copy)

save_path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\intersubj\corrmap\hipptc_regressed_on_s1_map.nii"

nib.save(new_mni, save_path)
