"""2021-08-09-gifify.py

Creates animated GIFs of progress-tracking images
"""


from pathlib import Path
from pyprojroot import here
import imageio
import parse



folders_to_process = [
    here("plots/2020-11-05-simple-mfbo"),
    here("plots/2021-07-06-manual-additions"),
]


img_name_template = parse.compile("{name_spec}-{budget:d}.png")





for folder in folders_to_process:
    """
    get set/list of 'name_specs' that have to be processed

    for name_spec to process:
        create output-name
        get all relevant input files
        sort in correct order

        create writer and make GIF
    """
    print(f"processing folder {folder.name}")
    pngs_in_folder = [fname for fname in folder.iterdir() if img_name_template.parse(fname.name)]
    spec_set = set(img_name_template.parse(fname.name)['name_spec'] for fname in pngs_in_folder)

    for name_spec in spec_set:
        out_name = folder / f"{name_spec}.gif"
        print(f"    creating {out_name.name}")
        matching_files = [fname for fname in folder.iterdir() if name_spec in fname.name and 'png' in fname.name]
        matching_files.sort(key=lambda f: img_name_template.parse(f.name)['budget'], reverse=True)

        with imageio.get_writer(out_name, mode='I', duration=0.5) as writer:
            for filename in matching_files:
                image = imageio.imread(filename)
                writer.append_data(image)
