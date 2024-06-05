import os, sys
import time as tm
import subprocess as sbp
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def cleanScreen():
    """
    """
    p = sys.platform
    if 'win' in p:
        os.system('cls')
    elif 'linux' in p:
        os.system('clear')


def createDirs(dname):
    """
    """
    if not os.path.isdir(dname):
        os.makedirs(dname)


def listFnames(dirName,ext=''):
    """
    """
    fnames = []
    for (rootDir,dirList,filesList) in os.walk(dirName):
        fnames += [os.path.join(rootDir,f) for f in filesList if f.endswith(ext)]

    return fnames


def runProcess(loudnessExe, method, soundField, audioFile, refFile, refLevel, module='subprocess'):
    """
    """
    print(' |- running ISO_532-1 loudness calculation')

    if module == 'os':
        command = '{:s} {:s} {:s} \"{:s}\" \"{:s}\" {:d}'.format(loudnessExe, method, soundField, audioFile, refFile,
                                                                 refLevel)
        os.system(command)
        output = ''
    elif module == 'subprocess':
        comList = [loudnessExe, method, soundField, audioFile, refFile, str(refLevel)]  # subprocess module command list
        proc = sbp.Popen(comList, stdout=sbp.PIPE)
        output = proc.stdout.read()
    else:
        return NotImplementedError

    if 'error' in output:
        print(' |- ERROR: \n{}'.format(output))
        raise ValueError


def getData():
    """
    """
    print(' |- getting loudness features')

    dfLoudness = pd.read_csv('Loudness.csv', sep=';', skiprows=5, index_col=0)

    loud = dfLoudness.as_matrix()

    return loud


def saveData(fileDir, loud):
    """
    """
    print(' |- saving loudness features')

    file = fileDir

    np.save(file, loud)


def removeTemp():
    """
    """
    print(' |- removing temporary csv files')

    os.remove('Loudness.csv')
    print('    |- removing Loudness.csv')
    os.remove('SpecLoudness.csv')
    print('    |- removing SpecLoudness.csv')


if __name__ == '__main__':

    cleanScreen()

    # --- Parameters ---
    METHOD_DICT = {'Varying': 'Time_varying', 'Stationary': 'Stationary'}
    SF_DICT = {'Free': 'F', 'Diffuse': 'D'}

    # --- ISO 532-1 ---

    LOUDNESS_BINARY = os.path.join(os.getcwd(), 'ISO_532_bin', 'ISO_532-1.exe')

    METHOD = 'Varying'  # loudness calculation type (Stationary, Varying)
    SOUND_FIELD = 'Free'  # sound field type (Diffuse, Free)

    CAL_FILE = os.path.join(os.getcwd(), 'calibration_audio_file', 'calibration_signal_sine_1kHz_60dB.wav')
    RMS_DB = 60  # RMS of calibration signal

    # --- Data ---
    exe = LOUDNESS_BINARY

    meth = METHOD_DICT[METHOD]
    sf = SF_DICT[SOUND_FIELD]

    rfile = CAL_FILE
    rlev = RMS_DB

    input_dir = os.path.join(os.getcwd(), 'Dataset_wav')
    output_dir = input_dir + '_loudness'
    createDirs(output_dir)

    # --- Analysis ---

    audioFiles = listFnames(input_dir, 'wav')

    for audioFile in audioFiles:

        audioName = os.path.basename(audioFile).split('.')[0]

        fileDir = os.path.join(output_dir, audioName + '.npy')

        print('Processing the audio clip: \"{}\"'.format(audioName))
        st = tm.clock()

        try:
            runProcess(exe, meth, sf, audioFile, rfile, rlev)
        except ValueError:
            continue

        loud = getData()
        saveData(fileDir, loud)

        removeTemp()

        et = tm.clock()
        print('Loudness extraction is done in {:.3f} seconds.'.format(et - st))
