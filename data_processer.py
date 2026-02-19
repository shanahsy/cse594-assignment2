"""
Converts json data into dataframe, with speaker, text, and split: 'train', 'dev', 'test'
"""

import requests
import json
import pandas as pd

seasons = {}        
season_ids = {}   


def load_data():
    BASE = "https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_{:02d}.json"

    for i in range(1, 11):   # seasons 1 → 10
        url = BASE.format(i)
        r = requests.get(url)
        data = json.loads(r.text)

        seasons[i] = data
        season_ids[i] = data["season_id"]

    return seasons, season_ids

def extract_utterances_from_seasons(main_chars, seasons):
    rows = []
    
    for season_num, season_data in seasons.items():
        # Split: seasons 1 - 8 is train, 9 is dev, 10 is test
        split = ""
        if season_num == 9:
            split = 'val'
        elif season_num == 10:
            split = 'test'
        else:
            split = 'train'

        episodes = season_data["episodes"]

        for ep in episodes:
            episode_id = ep["episode_id"]
            scenes = ep["scenes"]

            for scene_idx, scene in enumerate(scenes):
                
                utterances = scene["utterances"]

                for utt_idx, utt in enumerate(utterances):
                    speakers = utt.get("speakers", [])
                    if len(speakers) != 1:
                        continue

                    speaker = speakers[0]
                    if speaker not in main_chars:
                        continue
                    
                    text = utt.get("transcript", "").strip()
                

                    rows.append({
                        "speaker": speaker,
                        "text": text,
                        "split": split
                    })

    return pd.DataFrame(rows)

if __name__=="__main__":
    main_chars = [
    'Joey Tribbiani',
    'Monica Geller',
    'Chandler Bing',
    'Phoebe Buffay',
    'Rachel Green',
    'Ross Geller'
    ]

    seasons, season_ids = load_data()

    df_full = extract_utterances_from_seasons(main_chars, seasons)

    # Save to CSV
    df_full.to_csv('friends_data.csv', index=False)
    print(f"Saved {len(df_full)} utterances to friends_data.csv")

    print(len(df_full))
    print(df_full.head())

    # df_full = extract_utterances_from_seasons(main_chars, seasons)
    # df_full.to_csv('friends_data.csv', index=False)
    # print(len(df_full))


    







