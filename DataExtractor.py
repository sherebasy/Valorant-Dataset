import datetime
import os
import re
from heapq import nlargest

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import dehaze, is_allowed_specific_char, is_text_of_interest, postprocess
from paddleocr import PaddleOCR, draw_ocr


class DataExtractor:
    """
        Class used for extracting data from frames.

        frames: a list of the frames.
        e2e_model_dir: directory containing the PGNet weights.
        save_dir: where to save everything.
        x_dim: dimensions of the specific attribute.
        """

    def __init__(self, frames: [], save_dir: str, e2e_model_dir: str = 'models/e2e_server_pgnetA_infer', use_gpu=True,
                 economy_dim: dict = None, time_dim: dict = None, message_dim: dict = None, rounds_dim: dict = None,
                 shield_dim: dict = None, health_dim: dict = None):
        assert save_dir is not None, 'Please enter a saving directory!'
        if len(frames) == 0:
            raise ValueError('Frames parameter can not be empty!')

        self.frames = frames
        self.e2e_model_dir = e2e_model_dir
        self.save_dir = save_dir
        self.use_gpu = use_gpu
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=use_gpu, show_log=False, e2e_algorithm="PGNet",
                             e2e_model_dir=self.e2e_model_dir, e2e_pgnet_polygon=True)

        self.economy_dim = economy_dim
        if self.economy_dim is None:
            self.economy_dim = {'y': 610, 'h': 700, 'x': 1080, 'w': 1280}

        self.time_dim = time_dim
        if self.time_dim is None:
            self.time_dim = {'y': 0, 'h': 65, 'x': 560, 'w': 720}

        self.message_dim = message_dim
        if self.message_dim is None:
            self.message_dim = {'y': 40, 'h': 215, 'x': 510, 'w': 780}

        self.rounds_dim = rounds_dim
        if self.rounds_dim is None:
            self.rounds_dim = {'team': {'y': 20, 'h': 45, 'x': 525, 'w': 575},
                               'opponent': {'y': 20, 'h': 45, 'x': 700, 'w': 745}}

        self.health_dim = health_dim
        if self.health_dim is None:
            self.health_dim = {'y': 650, 'h': 720, 'x': 382, 'w': 500}

        self.shield_dim = shield_dim
        if self.shield_dim is None:
            self.shield_dim = {'y': 670, 'h': 700, 'x': 350, 'w': 382}

        self.economy, self.total_economy = self.get_economy()
        self.time = self.get_time()
        self.message = self.get_message()
        self.team_rounds = self.get_won_rounds(team='team')
        self.opponent_rounds = self.get_won_rounds(team='opponent')
        self.shield = self.get_shield()
        self.health = self.get_health()

        self.data = self.post_process()
        self.data.to_csv(os.path.join(save_dir, 'dataset.csv'))
        print('done!')

    def get_economy(self) -> ([], []):
        print('Extracting economy!')
        y, h, x, w = self.economy_dim['y'], self.economy_dim['h'], self.economy_dim['x'], self.economy_dim['w']

        ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=self.use_gpu, show_log=False, ocr_version='PP-OCR',
                        det_model_dir='models\\ch_ppocr_server_v2.0_det_infer',
                        rec_model_dir='models\\en_number_mobile_v2.0_rec_infer')

        image_paths = self.frames
        save_dir = os.path.join(self.save_dir, 'economy')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        def rec(img_: np.ndarray, first_try=True):
            crop_img = img_[y:h, x:w].copy()
            if not first_try:
                crop_img = dehaze(crop_img)

            cv2.imwrite(save_dir + f'/result_{i}.jpg', crop_img)
            result = ocr.ocr(save_dir + f'/result_{i}.jpg', cls=False)

            image_ = Image.open(save_dir + f'/result_{i}.jpg').convert('RGB')

            boxes_, txts_, scores_ = [], [], []
            for line in result:
                if line[1][1] > 0.9:  # consider only txts with confidence > 0.9
                    txt = re.sub("[,.+]", "", line[1][0])
                    if txt.lower() == 'o':
                        txt = '0'
                    txts_.append(txt)
                    boxes_.append(line[0])
                    scores_.append(line[1][1])
                else:
                    boxes_, txts_, scores_, t_ = [], [], [], None
                    break

            if len(txts_) > 0 and len(txts_) % 2 != 0:
                try:
                    temp = re.findall(r'\d+', txts_[-1])[0]
                    t_ = int(temp)
                except IndexError:
                    t_ = None

            elif len(txts_) == 1:
                try:
                    temp = re.findall(r'\d+', txts_[0])[0]
                    t_ = int(temp)
                except IndexError:
                    t_ = None
            else:
                boxes_, txts_, scores_, t_ = [], [], [], None

            im_show_ = draw_ocr(image_, boxes_, txts_, scores_, font_path='./fonts/simfang.ttf')
            im_show_ = Image.fromarray(im_show_)

            return txts_, scores_, im_show_, t_

        economy = []
        total = []
        for i, image in enumerate(tqdm(image_paths)):
            img = cv2.imread(image)
            txts, scores, im_show, t = rec(img)
            if len(txts) == 0:
                txts, scores, im_show, t = rec(img, first_try=False)
            total.append(t)
            economy.append(txts)
            im_show.save(save_dir + f'/result_{i}_paddle.jpg')

        return economy, total

    def get_time(self) -> []:
        print('Extracting time!')
        image_paths = self.frames
        save_dir = os.path.join(self.save_dir, 'time')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ocr = self.ocr
        y, h, x, w = self.time_dim['y'], self.time_dim['h'], self.time_dim['x'], self.time_dim['w']

        def rec(img_: np.ndarray, first_try=True):

            crop_img_ = img_[y:h, x:w].copy()

            if not first_try:
                crop_img_ = dehaze(crop_img_)

            crop_img_ = cv2.copyMakeBorder(crop_img_, 150, 150, 150, 150, cv2.BORDER_WRAP, value=[255, 255, 255])
            cv2.imwrite(save_dir + f'/result_{i}.jpg', crop_img_)
            result = ocr.ocr(save_dir + f'/result_{i}.jpg', cls=False)
            image_ = Image.open(save_dir + f'/result_{i}.jpg').convert('RGB')

            boxes_, txts_, scores_ = [], [], []
            for line in result:
                txt_ = line[1][0].replace(" ", "")
                if txt_ == '':
                    continue
                if is_allowed_specific_char(txt_, mode='numbers'):
                    if len(txt_) == 3:
                        try:
                            _ = int(txt_)  # txt_ is 3 numbers but, the detection algorithm did not capture the ':'
                            txt_ = txt_[0] + ':' + txt_[1:]  # e.g. 111 -> 1:11
                        except ValueError:
                            continue
                    elif (len(txt_) == 4) and (not txt_.__contains__(':')):
                        try:
                            _ = int(txt_)  # txt_ is 4 numbers but, the detection algorithm did not capture the '.'
                            txt_ = txt_[:2] + '.' + txt_[2:]  # e.g. 0763 -> 07.63
                        except ValueError:
                            continue
                    else:
                        continue  # random numbers are detected. Not of interest

                    try:
                        datetime.datetime.strptime(txt_, '%M:%S')  # text is a valid time
                    except ValueError:
                        continue

                    if ':' in txt_:
                        if txt_ > '1:39':
                            continue  # text can not be greater than '1:39'

                    txts_.append(txt_)
                    boxes_.append(line[0])
                    scores_.append(line[1][1])

            im_show_ = draw_ocr(image_, boxes_, txts_, scores_, font_path='./fonts/simfang.ttf')
            im_show_ = Image.fromarray(im_show_)

            return txts_, scores_, im_show_

        time = []
        for i, image in enumerate(tqdm(image_paths)):
            img = cv2.imread(image)
            txts, scores, im_show = rec(img)
            t = None
            if len(txts) == 0:
                txts, scores, im_show = rec(img, first_try=False)

            if len(txts) > 0:
                score_dict = {item: [] for item in
                              set(txts)}

                for txt, score in zip(txts, scores):
                    score_dict[txt].append(score)

                for k, v in score_dict.items():  # calculate the average
                    score_dict[k] = sum(v) / len(v)

                largest_i = nlargest(len(score_dict), score_dict, key=score_dict.get)[0]
                if score_dict[largest_i] > 0.89:
                    t = largest_i

            time.append(t)

            im_show.save(save_dir + f'/result_{i}_paddle.jpg')

        return time

    def get_message(self, texts_to_detect: [] = None) -> []:
        print('Extracting messages!')

        image_paths = self.frames
        save_dir = os.path.join(self.save_dir, 'message')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ocr = self.ocr
        y, h, x, w = self.message_dim['y'], self.message_dim['h'], self.message_dim['x'], self.message_dim['w']

        def rec(img_: np.ndarray):

            crop_img_ = img_[y:h, x:w].copy()
            cv2.imwrite(save_dir + f'/result_{i}.jpg', crop_img_)
            result = ocr.ocr(save_dir + f'/result_{i}.jpg', cls=False)
            image_ = Image.open(save_dir + f'/result_{i}.jpg').convert('RGB')

            boxes = [line[0] for line in result]
            txts_ = [line[1][0] for line in result]
            scores_ = [line[1][1] for line in result]

            im_show_ = draw_ocr(image_, boxes, txts_, scores_, font_path='./fonts/simfang.ttf')
            im_show_ = Image.fromarray(im_show_)

            return txts_, scores_, im_show_

        messages = []

        if texts_to_detect is None:
            texts_to_detect = ['buyphase', 'phase', 'spike', 'spikecarrierkilled', 'eleminated', 'teameleminated',
                               'spikedetonated', 'team', 'won', 'lost', 'defusing', 'defenders', 'attackers',
                               'flawless', 'clutch', 'round', 'spikeplanted', 'planted', 'thrifty']

        for i, image in enumerate(tqdm(image_paths)):
            img = cv2.imread(image)
            txts, scores, im_show = rec(img)
            m = is_text_of_interest(txts, texts_to_detect)
            messages.append(m)

            im_show.save(save_dir + f'/result_{i}_paddle.jpg')

        return messages

    def get_won_rounds(self, scale_percent: int = 75, team: str = None) -> []:
        image_paths = self.frames
        if team is None:
            raise ValueError("Choose either 'team' or 'opponent'")

        assert team in ['team', 'opponent'], f"team parameter has to be either 'team' or 'opponent'. Got {team}"

        ocr = self.ocr
        won_rounds = []
        if team == 'team':
            print('Extracting team won rounds!')
            y, h = self.rounds_dim['team']['y'], self.rounds_dim['team']['h']
            x, w = self.rounds_dim['team']['x'], self.rounds_dim['team']['w']
            save_dir = os.path.join(self.save_dir, 'won_rounds_team')
        else:
            print('Extracting opponent won rounds!')
            y, h = self.rounds_dim['opponent']['y'], self.rounds_dim['opponent']['h']
            x, w = self.rounds_dim['opponent']['x'], self.rounds_dim['opponent']['w']
            save_dir = os.path.join(self.save_dir, 'won_rounds_opponent')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        def rec(img__: np.ndarray, j: int):

            width = int(img__.shape[1] * scale_percent / 100)
            height = int(img__.shape[0] * scale_percent / 100)
            dim = (width, height)
            crop_img_ = img__[y:h, x:w].copy()

            crop_img_ = cv2.resize(crop_img_, dim, interpolation=cv2.INTER_LANCZOS4)
            if len(img__.shape) == 2:
                _, crop_img_ = cv2.threshold(crop_img_, 125, 100, cv2.THRESH_OTSU)
            crop_img_ = cv2.copyMakeBorder(crop_img_, 1100, 1100, 1100, 1100, cv2.BORDER_WRAP, value=[255, 255, 255])

            cv2.imwrite(save_dir + f'/result_{j}.jpg', crop_img_)
            result = ocr.ocr(save_dir + f'/result_{j}.jpg', cls=False)
            image_ = Image.open(save_dir + f'/result_{j}.jpg').convert('RGB')

            boxes, txts_, scores_ = [], [], []
            for line in result:
                txt_ = line[1][0].replace(" ", "")
                if is_allowed_specific_char(txt_, mode='numbers'):
                    try:
                        temp = int(txt_)
                        if temp > 13:  # maximum number of won rounds
                            continue
                    except ValueError:
                        continue

                    txts_.append(txt_)
                    boxes.append(line[0])
                    scores_.append(line[1][1])

            im_show_ = draw_ocr(image_, boxes, txts_, scores_, font_path='./fonts/simfang.ttf')
            im_show_ = Image.fromarray(im_show_)

            return txts_, scores_, im_show_

        for i, image in enumerate(tqdm(image_paths)):
            img = cv2.imread(image)
            img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            txts, scores, im_show = rec(img_, i)
            # if len(txts) == 0:  # no detection -> try image with colors
            #     txts, scores, im_show = rec(img, i)

            res = None
            if len(txts) > 0:
                score_dict = {item: [] for item in
                              set(txts)}

                for txt, score in zip(txts, scores):
                    score_dict[txt].append(score)

                for k, v in score_dict.items():  # calculate the average
                    score_dict[k] = sum(v) / len(v)

                largest_i = nlargest(len(score_dict), score_dict, key=score_dict.get)[0]
                if score_dict[largest_i] > 0.6:
                    res = largest_i

            won_rounds.append(res)
            im_show.save(save_dir + f'/result_{i}_paddle_{res}.jpg')

        return won_rounds

    def get_shield(self, scale_percent: int = 25) -> []:
        print('Extracting shield!')

        image_paths = self.frames
        ocr = self.ocr

        shield = []
        y, h, x, w = self.shield_dim['y'], self.shield_dim['h'], self.shield_dim['x'], self.shield_dim['w']

        save_dir = os.path.join(self.save_dir, 'shield')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        def rec(img__: np.ndarray, j: int):

            width = int(img__.shape[1] * scale_percent / 100)
            height = int(img__.shape[0] * scale_percent / 100)
            dim = (width, height)
            crop_img = img__[y:h, x:w].copy()

            crop_img = cv2.resize(crop_img, dim, interpolation=cv2.INTER_LANCZOS4)
            if len(img__.shape) == 2:
                _, crop_img = cv2.threshold(crop_img, 125, 100, cv2.THRESH_OTSU)
            crop_img = cv2.copyMakeBorder(crop_img, 200, 200, 200, 200, cv2.BORDER_WRAP, value=[255, 255, 255])

            cv2.imwrite(save_dir + f'/result_{j}.jpg', crop_img)
            result = ocr.ocr(save_dir + f'/result_{j}.jpg', cls=False)
            image_ = Image.open(save_dir + f'/result_{j}.jpg').convert('RGB')

            boxes_, txts_, scores_ = [], [], []
            for line in result:
                txt_ = line[1][0].replace(" ", "")
                if is_allowed_specific_char(txt_, mode='numbers'):
                    try:
                        temp = int(txt_)
                        if temp > 100:  # maximum shield health
                            continue
                    except ValueError:
                        continue

                    txts_.append(txt_)
                    boxes_.append(line[0])
                    scores_.append(line[1][1])

            im_show_ = draw_ocr(image_, boxes_, txts_, scores_, font_path='./fonts/simfang.ttf')
            im_show_ = Image.fromarray(im_show_)

            return txts_, scores_, im_show_

        for i, image in enumerate(tqdm(image_paths)):
            img = cv2.imread(image)
            img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            txts, scores, im_show = rec(img_, i)

            res = None
            if len(txts) > 0:
                score_dict = {item: [] for item in
                              set(txts)}

                for txt, score in zip(txts, scores):
                    score_dict[txt].append(score)

                for k, v in score_dict.items():  # calculate the average
                    score_dict[k] = sum(v) / len(v)

                largest_i = nlargest(len(score_dict), score_dict, key=score_dict.get)[0]
                if score_dict[largest_i] > 0.6:
                    res = largest_i

            shield.append(res)
            im_show.save(save_dir + f'/result_{i}_paddle_{res}.jpg')

        return shield

    def get_health(self, scale_percent: int = 75) -> []:
        print('Extracting health!')

        image_paths = self.frames
        ocr = self.ocr

        health = []
        y, h, x, w = self.health_dim['y'], self.health_dim['h'], self.health_dim['x'], self.health_dim['w']

        save_dir = os.path.join(self.save_dir, 'health')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        def rec(img__: np.ndarray, j: int):

            width = int(img__.shape[1] * scale_percent / 100)
            height = int(img__.shape[0] * scale_percent / 100)
            dim = (width, height)
            crop_img = img__[y:h, x:w].copy()

            crop_img = cv2.resize(crop_img, dim, interpolation=cv2.INTER_LANCZOS4)
            if len(img__.shape) == 2:
                _, crop_img = cv2.threshold(crop_img, 125, 100, cv2.THRESH_OTSU)
            crop_img = cv2.copyMakeBorder(crop_img, 500, 500, 500, 500, cv2.BORDER_WRAP, value=[255, 255, 255])

            cv2.imwrite(save_dir + f'/result_{j}.jpg', crop_img)
            result = ocr.ocr(save_dir + f'/result_{j}.jpg', cls=False)
            image_ = Image.open(save_dir + f'/result_{j}.jpg').convert('RGB')

            boxes_, txts_, scores_ = [], [], []
            for line in result:
                txt_ = line[1][0].replace(" ", "")
                if is_allowed_specific_char(txt_, mode='numbers'):
                    try:
                        temp = int(txt_)
                        if temp > 100:  # maximum shield health
                            continue
                    except ValueError:
                        continue

                    txts_.append(txt_)
                    boxes_.append(line[0])
                    scores_.append(line[1][1])

            im_show_ = draw_ocr(image_, boxes_, txts_, scores_, font_path='./fonts/simfang.ttf')
            im_show_ = Image.fromarray(im_show_)

            return txts_, scores_, im_show_

        for i, image in enumerate(tqdm(image_paths)):
            img = cv2.imread(image)
            img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            txts, scores, im_show = rec(img_, i)
            # if len(txts) == 0:  # no detection -> try image with colors
            #     txts, scores, im_show = rec(img, i)

            res = None
            if len(txts) > 0:
                score_dict = {item: [] for item in
                              set(txts)}

                for txt, score in zip(txts, scores):
                    score_dict[txt].append(score)

                for k, v in score_dict.items():  # calculate the average
                    score_dict[k] = sum(v) / len(v)

                largest_i = nlargest(len(score_dict), score_dict, key=score_dict.get)[0]
                if score_dict[largest_i] > 0.6:
                    res = largest_i

            health.append(res)
            im_show.save(save_dir + f'/result_{i}_paddle_{res}.jpg')

        return health

    def post_process(self):
        print('Postprocessing!')
        data = postprocess(time=self.time, messages=self.message, team_rounds=self.team_rounds,
                           opponent_rounds=self.opponent_rounds, economy=self.economy, total=self.total_economy,
                           shield=self.shield, health=self.health)

        return data
