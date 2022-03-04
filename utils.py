import datetime
import re
from difflib import SequenceMatcher
import cv2
import numpy as np
import pandas as pd

from guidedfilter import guided_filter
from pytube import YouTube


def filter_message(m):
    texts_to_detect = {
        'buy': ['buyphase', 'phase'],
        'spike_planted': ['spikeplanted', 'planted', 'lanted'],
        'round_end': ['round', 'eleminated', 'teameleminated', 'spikedetonated', 'detonated', 'team', 'won', 'lost',
                      'defused', 'defenders', 'attackers', 'flawless', 'clutch', 'thrifty']
    }
    if m is None:
        return ''
    elif is_text_of_interest(m, texts_to_detect['buy']):
        return 'BUY'
    elif is_text_of_interest(m, texts_to_detect['spike_planted']):
        return 'SPIKE PLANTED'
    elif is_text_of_interest(m, texts_to_detect['round_end']):
        return 'ROUND END'
    else:
        return ''


def timedelta_to_time(x):
    if pd.isnull(x):
        return pd.NaT
    else:
        seconds = x.seconds
        minutes = (seconds // 60) % 60
        return datetime.time(minute=minutes, second=seconds - minutes * 60).strftime("%M:%S")


def postprocess(time: [] = None, messages: [] = None, team_rounds: [] = None, opponent_rounds: [] = None,
                economy: [] = None, total: [] = None, shield: [] = None, health: [] = None):
    data = pd.DataFrame(
        {'time': time,
         'message': messages,
         'team_rounds': team_rounds,
         'opponent_rounds': opponent_rounds,
         'economy': economy,
         'economy_total': total,
         'shield': shield,
         'health': health
         })

    columns = ['team_rounds', 'opponent_rounds', 'shield', 'health']
    data[columns] = data[columns].fillna(-1)
    data[columns] = data[columns].astype(int)
    data[columns] = data[columns].astype(str)
    data[columns] = data[columns].replace('-1', np.nan)
    data[columns] = data[columns].astype(float)

    data['phase'] = data.message.apply(lambda x: filter_message(x))
    data.time.fillna('', inplace=True)
    data.time = data.time.apply(
        lambda x: datetime.timedelta(minutes=int(x.split(':')[0]), seconds=int(x.split(':')[1])) if ':' in x else x)

    for index, row in data.iterrows():
        phases = ['SPIKE PLANTED', 'ROUND END']
        if data.loc[index, 'phase'] in phases:
            data.loc[index, 'time'] = pd.NaT
            continue

        elif pd.isnull(row['time']):
            t = data.loc[index - 1, 'time'] - datetime.timedelta(seconds=1)
            if t != datetime.timedelta(days=-1, hours=23, minutes=59, seconds=59):
                data.loc[index, 'time'] = t

        if (data.loc[index, 'time'] > datetime.timedelta(minutes=0, seconds=29)) and (data.loc[index, 'phase'] == ''):
            data.loc[index, 'phase'] = 'Normal'

        if (data.loc[index, 'phase'] == 'Normal') and (data.loc[index + 1, 'phase'] not in phases + ['BUY']):
            data.loc[index + 1, 'phase'] = 'Normal'

    indices = data[(~pd.isnull(data.time)) & (data.phase == '')].index
    data.loc[indices, 'phase'] = 'BUY'

    indices = data[data.phase == 'ROUND END'].index
    for v, w in zip(indices[::1], indices[1::1]):
        if v == indices[0]:
            slice_df = data.loc[:v - 1]
        else:
            slice_df = data.loc[v + 1:w - 1]

        i = slice_df.index

        mode_team = slice_df.loc[i, 'team_rounds'].dropna().mode()
        mode_opp = slice_df.loc[i, 'opponent_rounds'].dropna().mode()

        if len(mode_team) != 0:
            slice_df.loc[i, 'team_rounds'] = mode_team[0]
        if len(mode_opp) != 0:
            slice_df.loc[i, 'opponent_rounds'] = mode_opp[0]

    data.loc[data.phase == 'ROUND END', ['team_rounds', 'opponent_rounds']] = np.nan
    data['team_rounds'].fillna(method='bfill', inplace=True)
    data['opponent_rounds'].fillna(method='bfill', inplace=True)

    data['team_rounds'].fillna(method='ffill', inplace=True)
    data['opponent_rounds'].fillna(method='ffill', inplace=True)

    indices = data[(data.phase == 'BUY') | (data.economy.str.len() > 1)].index
    for v, w in zip(indices[::1], indices[1::1]):

        if v == indices[0]:
            slice_df = data.loc[:v - 1]
        else:
            slice_df = data.loc[v + 1:w - 1]

        i = slice_df.index

        mode_economy_total = slice_df.loc[i, 'economy_total'].dropna().mode()

        if len(mode_economy_total) != 0:
            slice_df.loc[i, 'economy_total'] = mode_economy_total[0]

    # indices = data[(data.phase == 'BUY') | (data.economy.str.len() > 1)].index

    data.time = data.time.apply(lambda x: timedelta_to_time(x))

    return data


def is_text_of_interest(txts, texts_to_detect):
    """
    Give a list of texts, decides whether any of the elements of texts_to_detect are available in txts.
    :param txts: list of texts
    :param texts_to_detect: list of texts to detect
    :return: txts if there is a similarity. None if no similarity is found
    """
    for txt in txts:
        for txt_to_det in texts_to_detect:
            txt = txt.replace(" ", "").lower()
            if (len(set(txt) & set(txt_to_det)) >= 0.82 * len(txt_to_det)) | (similar(txt, txt_to_det) == 1):
                if similar(txt, txt_to_det) > 0.5:
                    return txts
    return None


def download_youtube_video(url, filename, path):
    """
    Given an url, a saving directory, and an index i, downloads a YouTube video to saving directory with the name
    filename
    :param filename: title of the video to save it as
    :param url: link of the YouTube video
    :param path: location where to save the video

    :returns vid_name: name of the saved video
    """
    youtube_video = YouTube(url)
    video = youtube_video.streams.get_highest_resolution()
    print(f"Downloading Video: {video.title}")
    video.download(path, filename=filename)
    print("Finished Downloading")


def is_allowed_specific_char(string: str, mode: str = 'all') -> bool:
    """
    Given a string, determines if it contains ONLY characters from a set of numbers (mode = 'numbers') or a set of
    numbers and alphabet (mode = 'all'.)

    :param string: string to be analyzed
    :param mode: either 'all' or 'numbers'
    :return: boolean
    """
    assert string is not None, 'Please enter a string!'
    assert len(string) > 0, 'Please enter a string!'
    assert mode.lower() in ['all', 'numbers'], f"mode has to be 'numbers' or 'all'. Got {mode}!"
    re_statement = r'[^a-zA-Z0-9 ]'
    if mode == 'numbers':
        re_statement = r'[^0-9:. ]'
    charRe = re.compile(re_statement)
    string = charRe.search(string)

    return not bool(string)


def similar(a: str, b: str) -> float:
    """
    Given string a & b, returns the ratio of similarity between both strings. Disregards uppercase letters.
    """
    a = a.lower()
    b = b.lower()
    return SequenceMatcher(None, a, b).ratio()


# The following functions are used from
# https://github.com/spmallick/learnopencv/tree/master/Improving-Illumination-in-Night-Time-Images


tmin = 0.1  # minimum value for t to make J image
w = 15  # window size, which determine the corseness of prior images
alpha = 0.4  # threshold for transmission correction
omega = 0.75  # this is for dark channel prior
p = 0.1  # percentage to consider for atmosphere
eps = 1e-3  # for J image


def get_illumination_channel(I, w=w):
    M, N, _ = I.shape
    padded = np.pad(I, ((int(w / 2), int(w / 2)), (int(w / 2), int(w / 2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])

    return darkch, brightch


def get_atmosphere(I, brightch, p=0.1):
    M, N = brightch.shape
    flatI = I.reshape(M * N, 3)
    flatbright = brightch.ravel()

    searchidx = (-flatbright).argsort()[:int(M * N * p)]
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A


def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch - A_c) / (1. - A_c)
    return (init_t - np.min(init_t)) / (np.max(init_t) - np.min(init_t))


def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha=alpha, omega=omega, w=w):
    im3 = np.empty(I.shape, I.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = I[:, :, ind] / A[ind]
    dark_c, _ = get_illumination_channel(im3, w)
    dark_t = 1 - omega * dark_c
    corrected_t = init_t
    diffch = brightch - darkch

    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if diffch[i, j] < alpha:
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

    return np.abs(corrected_t)


def get_final_image(I, A, refined_t, tmin=tmin):
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3))
    J = (I - A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A

    return (J - np.min(J)) / (np.max(J) - np.min(J))


def dehaze(I, tmin=tmin, w=w, alpha=alpha, omega=omega, p=p, eps=eps, reduce=False):
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)

    init_t = get_initial_transmission(A, Ibright)
    if reduce:
        init_t = reduce_init_t(init_t)
    corrected_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, alpha, omega, w)

    normI = (I - I.min()) / (I.max() - I.min())
    refined_t = guided_filter(normI, corrected_t, w, eps)
    J_refined = get_final_image(I, A, refined_t, tmin)

    enhanced = (J_refined * 255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced


def reduce_init_t(init_t):
    init_t = (init_t * 255).astype(np.uint8)
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    init_t = cv2.LUT(init_t, table)
    init_t = init_t.astype(np.float64) / 255
    return init_t
