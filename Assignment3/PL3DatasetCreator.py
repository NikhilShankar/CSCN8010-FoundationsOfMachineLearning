import os, shutil, pathlib


class PL3DatasetCreator:

    def __init__(self, originaldir= pathlib.Path("data/RawDataset"),
                                      newdirName = "PL3-Dataset"): 
        self.original_dir = originaldir
        self.new_base_dir = pathlib.Path(f"data/{newdirName}")
    

    def make_subset(self, subset_name, start_index, end_index):
        for category in ("Cat", "Dog"):
            dir = self.new_base_dir / subset_name / category
            os.makedirs(dir)
            fnames = [f"/{category}/{i}.jpg" for i in range(start_index, end_index)]
            fname = [f"{category}{i}.jpg" for i in range(start_index, end_index)]
            for index in range(0, len(fnames)):
                shutil.copyfile(src=f'{self.original_dir}/{fnames[index]}',
                                dst=f'{dir}/{fname[index]}')


    def createDataset(self, startIndex, total, trainPercent, validationPercent):
        trainEndIndex = startIndex + total//(100//trainPercent)
        validationEndIndex = trainEndIndex + total//(100//validationPercent)
        testEndIndex = validationEndIndex + total//(100//(100-(trainPercent + validationPercent)))
        print(f'{trainEndIndex} {validationEndIndex} {testEndIndex}')
        self.make_subset("train", start_index=startIndex, end_index= trainEndIndex)
        self.make_subset("validation", start_index=trainEndIndex, end_index=validationEndIndex)
        self.make_subset("test", start_index=validationEndIndex, end_index=testEndIndex)