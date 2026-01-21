# Import dependencies
import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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

def return_paths(roi, subject):
    movie_path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\subjects\s" + str(subject) + r"\sherlock_movie" + roi + r"_sherlock_movie_s" + str(subject) + r".mat"
    recall_path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\subjects\s" + str(subject) + r"\sherlock_recall" + roi + r"_sherlock_recall_s" + str(subject) + r".mat"
    return (movie_path, recall_path)

def test_shape(lst):
    print(lst)
    print()
    for x in lst:
        try:
            print(x.shape)
        except:
            print("list elements are not numpy arrays")

# Define dataset parameters (number of subjects, roi, etc...)
rdata_movie_ls = []
rdata_recall_ls = []
# roi_chosen = r"\aud_early"
roi_chosen = r"\pmc_nn"
num_subj = 17
custom_palette = matplotlib.colors.LinearSegmentedColormap.from_list("my_blend", ["black", "red", "orange", "yellow", "white"])

# Extract fMRI data for each subject
for x in range(1, num_subj+1):
    (mv_path, rcl_path) = return_paths(roi_chosen, x)
    mv_data = loadmat(mv_path)
    rcl_data = loadmat(rcl_path)
    rdata_movie_ls.append(mv_data["rdata"])
    rdata_recall_ls.append(rcl_data["rdata"])

# Extract TR
num_TR = len(rdata_movie_ls[0][0])

roitc_temp = []

# Compute mean voxel activity at each TR for each subject
for subject_number in range(num_subj):
    rdata = rdata_movie_ls[subject_number]
    sum_rdata = np.sum(rdata, axis=0)/num_TR
    roitc_temp.append(sum_rdata)

# Organize data
roitc_untransposed = np.array(roitc_temp)
roitc = np.matrix.transpose(roitc_untransposed)

# Compute mean voxel activity at each TR across all subjects
roitc_mean = np.sum(roitc, axis=1)/(num_subj)

# Plot mean voxel activity per subject + across all subjects
colorlist = sns.color_palette("husl", num_subj)
labels = ["s"+str(x+1) for x in range(num_subj)] + ["mean"]

for iter in range(num_subj):
    plt.plot(roitc_untransposed[iter], color=colorlist[iter])

plt.plot(roitc_mean, color="black")
plt.xlabel("TR")
plt.ylabel("Z")
plt.legend(labels)
plt.show()
plt.clf()

# Compute lagged correlation between each subject and average of other subjects
set_lag = 50
lag_corr_ls = []
lags = np.arange(-set_lag, set_lag + 1)

for curr_sub in range(num_subj):
    curr_roitc = roitc_untransposed[curr_sub]
    mean_roitc_curr_excl = (roitc_mean*num_subj - curr_roitc)/(num_subj-1)
    lag_corr = compute_lagged_corr(curr_roitc, mean_roitc_curr_excl, set_lag)
    lag_corr_ls.append(lag_corr)

# Plot lagged correlations
for iter in range(num_subj):
    plt.plot(lag_corr_ls[iter], color=colorlist[iter])

labels.pop(-1)
plt.xlabel("TR")
plt.ylabel("Z")
plt.legend(labels)
plt.show()
plt.clf()

# Compute and plot timepoint-by-timepoint cross-subject pattern correlation matrices for movie-movie
corr_mat_ls = []
rdata_sum = sum(rdata_movie_ls)

for curr_sub in range(num_subj):
    curr_data = rdata_movie_ls[curr_sub]
    other_subj_data = (rdata_sum-curr_data)/(num_subj-1)
    corr_mat = corr_inter(other_subj_data, curr_data)
    corr_mat_ls.append(corr_mat)

avg_corr_mat = sum(corr_mat_ls)/num_subj
avg_corr_mat_diag = np.diag(avg_corr_mat)

ticks = [x for x in range(0, num_TR, 200)]
tick_labels = [str(x) for x in ticks]
avg_corr_mat_heatmap = sns.heatmap(avg_corr_mat, vmin=-0.3, vmax=0.3, cmap=custom_palette)
avg_corr_mat_heatmap.set(xlabel="Time (TR)", ylabel="Time (TR)",
                         xticks=ticks, xticklabels=tick_labels,
                         yticks=ticks, yticklabels=tick_labels)
plt.show()
plt.clf()

# Plot recall behavior for a specific subject
subj_chosen = 1
subj_chosen -= 1
path = r"C:\Users\vmiss\Documents\JChenLab\Research\Computational_Neuroscience\sherlock_nifti_kit_v2_withdata\subjects\sherlock_allsubs_events.mat"
ev = loadmat(path)["ev"][0, 0]
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

plt.imshow(recall_behavior_mat, aspect="auto", cmap='Greys_r')
plt.title("Subject " + str(subj_chosen+1))
plt.xlabel("Recall time (TRs)")
plt.ylabel("Movie time (TRs)")
plt.show()
plt.clf()

# Compute matrices to find voxel pattern averages for each scene
names = ["s"+str(x+1) for x in range(num_subj)]
count = 0
roi_chosen = r"\pmc_nn"
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
    fullSL1 = subj_data["fullSL1"]
    fullSL2 = subj_data["fullSL2"] + fullSL1[-1, -1]
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

# Start computing scene-level pattern correlations
num_subj = 17
nEvents = all_subs_50M_events.shape[1]
sdiag_movrec_withinsubj = []
mdiag_movrec_withinsubj = []
movrec_withinsubj_corrmat = []
movmov_btwnsubj_corrmat = []
sdiag_movmov_btwnsubj = []
mdiag_movmov_btwnsubj = []
recrec_btwnsubj_corrmat = []
sdiag_recrec_btwnsubj = []
mdiag_recrec_btwnsubj = []

for n in range(num_subj):
    sdiag_movrec_withinsubj.append(np.diagonal(corr_inter(all_subs_iM_events[n], all_subs_iFr_events[n])))
    mdiag_movrec_withinsubj.append(np.nanmean(sdiag_movrec_withinsubj[n]))

    movrec_withinsubj_corrmat.append(corr_inter(all_subs_50Fr_events[:,:,n], all_subs_50M_events[:,:,n]))

    others = [i for i in range(num_subj) if i != n]
    movmov_btwnsubj_corrmat.append(corr_inter(all_subs_50M_events[:,:,n], np.nanmean(all_subs_50M_events[:,:,others], axis=2)))
    sdiag_movmov_btwnsubj.append(np.diagonal(corr_inter(all_subs_50M_events[:,:,n], np.nanmean(all_subs_50M_events[:,:,others], axis=2))))
    mdiag_movmov_btwnsubj.append(np.nanmean(sdiag_movmov_btwnsubj[n]))

    recrec_btwnsubj_corrmat.append(corr_inter(all_subs_50Fr_events[:,:,n], np.nanmean(all_subs_50Fr_events[:,:,others], axis=2)))
    sdiag_recrec_btwnsubj.append(np.diagonal(corr_inter(all_subs_50Fr_events[:,:,n], np.nanmean(all_subs_50Fr_events[:,:,others], axis=2))))
    mdiag_recrec_btwnsubj.append(np.nanmean(sdiag_recrec_btwnsubj[n]))

movrec_withinsubj_corrmat = np.transpose(np.array(movrec_withinsubj_corrmat), (1, 2, 0))
movmov_btwnsubj_corrmat = np.transpose(np.array(movmov_btwnsubj_corrmat), (1, 2, 0))
recrec_btwnsubj_corrmat = np.transpose(np.array(recrec_btwnsubj_corrmat), (1, 2, 0))

# Plot the relevant graphs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ticks = [x for x in range(0, movrec_withinsubj_corrmat.shape[0], int(movrec_withinsubj_corrmat.shape[0]/10))]
tick_labels = [str(x) for x in ticks]

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

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(mdiag_movrec_withinsubj, ax=axes[0])
axes[0].set(title="Spatial Corr Within Subject Movie vs. Recall", xlabel="Subject", ylabel="R")

sns.barplot(mdiag_movmov_btwnsubj, ax=axes[1])
axes[1].set(title="Spatial Corr Between Subjects Movie vs. Movie", xlabel="Subject", ylabel="R")

sns.barplot(mdiag_recrec_btwnsubj, ax=axes[2])
axes[2].set(title="Spatial Corr Between Subjects Recall vs. Recall", xlabel="Subject", ylabel="R")

plt.show()

plt.clf()
