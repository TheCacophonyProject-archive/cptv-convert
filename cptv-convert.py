import argparse
import os
import pickle
import numpy as np
import PIL as pillow
import tempfile
from mpeg_creator import MPEGCreator
from PIL import Image
from cptv import CPTVReader
from os.path import isfile, join
from shutil import copyfile

COLORMAP_FILE = join(os.path.dirname(os.path.realpath(__file__)), "colormap.dat")

def process_cptv_file(cptv_file, output_folder, copy, delete, colormap):
    with open(cptv_file, "rb") as f:
        reader = CPTVReader(f)

        # make name and folder for recording
        fileName = reader.timestamp.strftime("%Y%m%d-%H%M%S")
        raw_name = fileName + ".cptv"
        mp4_name = fileName + ".mp4"
        mp4_temp = join(tempfile.gettempdir(), mp4_name)
        devicename = reader.device_name
        if devicename == None or devicename == "":
            devicename = "NO_DEVICE_NAME"
        else:
            raw_name = devicename + "_" + raw_name
            mp4_name = devicename + "_" + mp4_name
        raw_dir = join(output_folder, devicename, "raw-cptv")
        mp4_dir = join(output_folder, devicename, "cptv-processed")
        os.makedirs(mp4_dir, exist_ok=True)
        raw_file = join(raw_dir, raw_name)
        mp4_file = join(mp4_dir, mp4_name)
        print(mp4_file)
        if copy:
            os.makedirs(raw_dir, exist_ok=True)
            copy_file(cptv_file, raw_file)

        FRAME_SCALE = 4.0
        NORMALISATION_SMOOTH = 0.95
        HEAD_ROOM = 25
        auto_min = None
        auto_max = None

        mpeg = MPEGCreator(mp4_temp)
        for frame in reader:
            if (auto_min == None):
                auto_min = np.min(frame[0])
                auto_max = np.max(frame[0])

            thermal_min = np.min(frame[0])
            thermal_max = np.max(frame[0])

            auto_min = NORMALISATION_SMOOTH * auto_min + (1 - NORMALISATION_SMOOTH) * (thermal_min-HEAD_ROOM)
            auto_max = NORMALISATION_SMOOTH * auto_max + (1 - NORMALISATION_SMOOTH) * (thermal_max+HEAD_ROOM)

            # sometimes we get an extreme value that throws off the autonormalisation, so if there are values outside
            # of the expected range just instantly switch levels
            if thermal_min < auto_min or thermal_max > auto_max:
                auto_min = thermal_min
                auto_max = thermal_max

            thermal_image = convert_heat_to_img(frame[0], colormap, auto_min, auto_max)
            thermal_image = thermal_image.resize((int(thermal_image.width * FRAME_SCALE), int(thermal_image.height * FRAME_SCALE)), Image.BILINEAR)
            mpeg.next_frame(np.asarray(thermal_image))
        mpeg.close()
    copy_file(mp4_temp, mp4_file)
    os.remove(mp4_temp)
    if copy and delete:
        os.remove(cptv_file)

def copy_file(src, dest):
    copyfile(src, dest)
    fs = os.open(dest, os.O_RDONLY)
    os.fsync(fs)
    os.close(fs)

def convert_heat_to_img(frame, colormap, temp_min = 2800, temp_max = 4200):
    """
    Converts a frame in float32 format to a PIL image in in uint8 format.
    :param frame: the numpy frame contining heat values to convert
    :param colormap: an optional colormap to use, if none is provided then tracker.colormap is used.
    :return: a pillow Image containing a colorised heatmap
    """
    # normalise
    frame = np.float32(frame)
    frame = (frame - temp_min) / (temp_max - temp_min)
    colorized = np.uint8(255.0*colormap(frame))
    img = pillow.Image.fromarray(colorized[:,:,:3]) #ignore alpha
    return img

def print_args(args):
    print("output folder: ", args.output_folder)
    print("source folder: ", args.source_folder)
    print("copy CPTV files: ", args.copy)
    print("delete origional CPTV: ", args.delete_origional)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('source_folder', help="Folder with cptv files to convert.")
    parser.add_argument('output_folder', help="Where to save mp4 files.")
    parser.add_argument('-c', '--copy', nargs='?', const=True, default=False, help="Will copy the raw recordings to the output folder.")
    parser.add_argument('-d', '--delete-origional', nargs='?', const=True, default=False, help="Will delete the origional recordings that were copied to the output folder.")
    parser.add_argument('-b', '--blink', nargs='?', const=True, default=False, help="Enable blinking (blinks green LED on RPi when converting and solid when done.)")
    parser.add_argument('--colormap', help="Colormap to use for conversion.")
    args = parser.parse_args()

    print("Processing CPTV files.")
    print_args(args)

    if args.blink:
        os.system("echo timer >/sys/class/leds/led0/trigger") #makes the green led blink on the RPi

    cptv_files = [join(args.source_folder, f) for f in os.listdir(args.source_folder) if isfile(join(args.source_folder, f))]
    print("files to convert: " + str(len(cptv_files)))
    if args.colormap == None:
        colormap = pickle.load(open(COLORMAP_FILE, 'rb'))
    else:
        colormap = pickle.load(open(args.colormap, 'rb'))

    for cptv_file in cptv_files:
        process_cptv_file(cptv_file, args.output_folder, args.copy, args.delete_origional, colormap)

    if args.blink:
        os.system("echo default-on >/sys/class/leds/led0/trigger")
    print("done")

if __name__ == "__main__":
    main()
