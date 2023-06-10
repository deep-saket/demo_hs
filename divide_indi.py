import sys
import os
import json

if __name__ == '__main__':
    json_path = sys.argv[1]

    with open(json_path) as f:
        data = json.load(f) 

        total = len(data['annotations'])
        categories = data['categories']
        info = data['info']

        for idx in range(total):
            image_info_dict = data['images'][idx]
            anno_info_dict = data['annotations'][idx]

            anno_name = os.path.basename(image_info_dict['file_name']).split('.')[0] + '.json'

            substance = {
                            'images' : [image_info_dict],
                            'categories' : [categories],
                            'annotations' : [anno_info_dict],
                            'info' : info
                        }

            print(anno_name)
            with open(anno_name, "w") as outfile:
                json.dump(substance, outfile)