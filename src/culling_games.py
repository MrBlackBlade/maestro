from miditok import REMI
from symusic import Score
from src.core.config import Config
from src.core.utils import get_tokenizer
# tokenizer = REMI(params="data/tokenizer.json")

tokenizer = get_tokenizer()

#vocab = tokenizer.vocab

position_token_ids = set()
pitch_token_ids = set()
program_token_ids = set() 
bar_token_ids = set()

for tok_str, tok_id in tokenizer.vocab.items():
    if "Position".lower() in tok_str.lower():
        position_token_ids.add(tok_id)
    elif "Pitch".lower() in tok_str.lower():
        pitch_token_ids.add(tok_id)
    elif "Program".lower() in tok_str.lower():
        program_token_ids.add(tok_id)
    elif "Bar".lower() in tok_str.lower():
        bar_token_ids.add(tok_id)

dict_decoder = {tok_id: tok_str for tok_str, tok_id in tokenizer.vocab.items()}
import json
with open('dict_decoder.json', 'w') as f:
    json.dump(dict_decoder, f, indent=4)

midi_path = Config.DATASETS_DIR / "XMIDI_angry_classical_0HP7PK58.mid"

midi = Score(midi_path)

tok_sequence = tokenizer(midi)

current_bar = set()
current_program = None
current_position = None
current_pitch = None

for token in tok_sequence.ids:
    if token in bar_token_ids:
        current_bar = set()
        # print("new bar")
    elif token in program_token_ids:
        current_program = token
    elif token in position_token_ids:
        current_position = token
    elif token in pitch_token_ids:
        current_pitch = token
        if current_program and current_position and current_pitch:
            note = (current_program, current_position, current_pitch)
            note_str = (dict_decoder[current_program], dict_decoder[current_position], dict_decoder[current_pitch])
            # print(note_str)
            if note in current_bar:
                print(f"Note {note} already in bar")
            else:
                current_bar.add(note)


