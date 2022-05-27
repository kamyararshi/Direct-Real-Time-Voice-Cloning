######### ----VC Libs---- #########
import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

import os
import argparse
from pathlib import Path
from utils.argutils import print_args

######### ----ASR Libs---- #########
import speech_recognition as sr
import time





parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-e", "--enc_model_fpath", type=Path,
                    default="saved_models/default/encoder.pt",
                    help="Path to a saved encoder")
parser.add_argument("-s", "--syn_model_fpath", type=Path,
                    default="saved_models/default/synthesizer.pt",
                    help="Path to a saved synthesizer")
parser.add_argument("-v", "--voc_model_fpath", type=Path,
                    default="saved_models/default/vocoder.pt",
                    help="Path to a saved vocoder")
parser.add_argument("--cpu", action="store_true", help=\
    "If True, processing is done on CPU, even when a GPU is available.")
parser.add_argument("--no_sound", action="store_true", help=\
    "If True, audio won't be played.")
parser.add_argument("--seed", type=int, default=None, help=\
    "Optional random number seed value to make toolbox deterministic.")
args = parser.parse_args()
arg_dict = vars(args)
print_args(args, parser)




## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
enc_pth = Path("saved_models/default/encoder.pt")
syn_pth = Path("saved_models/default/synthesizer.pt")
voc_pth = Path("saved_models/default/vocoder.pt")
ensure_default_models(Path("saved_models"))
encoder.load_model(enc_pth)
synthesizer = Synthesizer(syn_pth)
vocoder.load_model(voc_pth)




## Run a test
print("Testing your configuration with small inputs.")
# Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
# sampling rate, which may differ.
# If you're unfamiliar with digital audio, know that it is encoded as an array of floats
# (or sometimes integers, but mostly floats in this project) ranging from -1 to 1.
# The sampling rate is the number of values (samples) recorded per second, it is set to
# 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
# to an audio of 1 second.
print("\tTesting the encoder...")
encoder.embed_utterance(np.zeros(encoder.sampling_rate))

# Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
# returns, but here we're going to make one ourselves just for the sake of showing that it's
# possible.
embed = np.random.rand(speaker_embedding_size)
# Embeddings are L2-normalized (this isn't important here, but if you want to make your own
# embeddings it will be).
embed /= np.linalg.norm(embed)
# The synthesizer can handle multiple inputs with batching. Let's create another embedding to
# illustrate that
embeds = [embed, np.zeros(speaker_embedding_size)]
texts = ["test 1", "test 2"]
print("\tTesting the synthesizer... (loading the model will output a lot of text)")
mels = synthesizer.synthesize_spectrograms(texts, embeds)

# The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
# can concatenate the mel spectrograms to a single one.
mel = np.concatenate(mels, axis=1)
# The vocoder can take a callback function to display the generation. More on that later. For
# now we'll simply hide it like this:
no_action = lambda *args: None
print("\tTesting the vocoder...")
# For the sake of making this test short, we'll pass a short target length. The target length
# is the length of the wav segments that are processed in parallel. E.g. for audio sampled
# at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
# 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
# that has a detrimental effect on the quality of the audio. The default parameters are
# recommended in general.
vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

print("All test passed! You can now synthesize speech.\n\n")




print("Interactive generation loop")
num_generated = 0
while True:
    try:
        # Get the reference audio filepath
        message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                  "wav, m4a, flac, ...):\n"
        ##--------------Can Modify---------------##
        #Address of the desired sample voice to clone
        in_fpath = Path(r".\datasets_root\LibriSpeech\train-clean-100\19\227\19-227-0001.flac".replace("\"", "").replace("\'", ""))
        ##--------------         ----------------##
        
        ## Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is
        # important: there is preprocessing that must be applied.

        # The following two methods are equivalent:
        # - Directly load from the filepath:
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")

        # Then we derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")


        ##-------ASR Real-time-------##
        r = sr.Recognizer()
        r.energy_threshold = 150

        # Records your audio - stops when you no longer speak
        # TODO: Need a recorder that records with a frequency like every 5 sec
        # Then we feed every batch of 5 sec to the ASR and then VC on the flow
        # of the speech in real-time
        with sr.Microphone() as source:
            print("Say something!")
            tic = time.time()
            audio = r.listen(source)
            toc = time.time()

        print((toc-tic)//1, "s")

        ##-------Google's online API-------##
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)

            # Making the speech transcript using google's ASR API
            text = r.recognize_google(audio)
            print("Google Speech Recognition thinks you said:\n" + text)


            ##-------Voice Cloning-------##
            ## Generating the spectrogram
            # If seed is specified, reset torch seed and force synthesizer reload
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")


            ## Generating the waveform
            print("Synthesizing the waveform:")

            # If seed is specified, reset torch seed and reload vocoder
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)


            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.preprocess_wav(generated_wav)
        
        
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        
        
        # Play the audio (non-blocking)
        if not args.no_sound:
            import sounddevice as sd
            try:
                sd.stop()
                sd.play(generated_wav, synthesizer.sample_rate)
            except sd.PortAudioError as e:
                print("\nCaught exception: %s" % repr(e))
                print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
            except:
                raise

        # Save it on the disk
        filename = "demo_output_%02d.wav" % num_generated
        print(generated_wav.dtype)
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        print("\nSaved output as %s\n\n" % filename)


    except Exception as e:
        print("Caught exception: %s" % repr(e))
        raise ValueError
        print("Restarting\n")
