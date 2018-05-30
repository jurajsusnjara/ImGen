import os


class Mapping():
    def __init__(self):
        self.mapping = {}

    @staticmethod
    def get_tags_from(fname):
        tags = []
        with open(fname, 'r') as f:
            for line in f:
                tags.append(line.replace('\n', ''))
        return tags

    @staticmethod
    def get_id_from(fname):
        start_idx = 4
        end_idx = fname.find('.')
        return int(fname[start_idx:end_idx])

    @staticmethod
    def get_ids_from(fname):
        ids = []
        with open(fname, 'r') as f:
            for line in f:
                ids.append(int(line.replace('\n', '')))
        return ids

    def append_from_ann(self, fname, ann):
        ids = self.get_ids_from(fname)
        for id in ids:
            self.append2dict(id, [ann])

    def append2dict(self, id, tags):
        if self.mapping.get(id, None) is None:
            self.mapping[id] = tags
        else:
            self.mapping[id] += tags

    def extract_id2tags_mapping(self, tag_dir):
        for fname in os.listdir(tag_dir):
            full_path = tag_dir + '/' + fname
            tags = self.get_tags_from(full_path)
            id = self.get_id_from(fname)
            self.append2dict(id, tags)
        return self.mapping

    def extract_annotations(self, ann_dir):
        for fname in os.listdir(ann_dir):
            full_path = ann_dir + '/' + fname
            ann = fname[0:fname.find('.')]
            self.append_from_ann(full_path, ann)
        return self.mapping


tags_dir = '/home/juraj/Desktop/flickr_data/mirflickr/meta/tags'
ann_dir = '/home/juraj/Desktop/flickr_data/mirflickr25k_annotations_v080'
mapping = Mapping()
mapping.extract_annotations(ann_dir)
mapping.extract_id2tags_mapping(tags_dir)

# TODO duplikati
