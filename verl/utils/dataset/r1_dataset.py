import pandas as pd
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

PROMPT_TEMPLATE = ("A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
                   "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
                   "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
                   "User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. "
                   "And your final answer will be extracted automatically by the \\boxed{{}} tag. {prompt}\n"
                   "Assistant: <think>")

class R1Dataset(RLHFDataset):

    def _wrap_prompt(self, chat):
        for msg in chat:
            if msg['role'] == 'user':
                prompt = msg['content']
                return PROMPT_TEMPLATE.format(prompt=prompt)

        return ""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: 0 < len(
            tokenizer.encode(self._wrap_prompt(doc[prompt_key]))) <= self.max_prompt_length,
                                                             axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self, start_idx=0):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')
        start_idx = start_idx % len(self.dataframe)
        df = self.dataframe
        df = pd.concat([df[start_idx:], df[:start_idx]], axis=0).reset_index(drop=True)
        self.dataframe = df

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self._wrap_prompt(chat)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
