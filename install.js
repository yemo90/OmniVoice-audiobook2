module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Clone the OmniVoice source into the app folder
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/k2-fsa/OmniVoice.git app"
        ]
      }
    },
    // Installing the package pulls torch/torchaudio automatically.
    // The project's pyproject.toml routes them to the CUDA 12.8 index on Windows/Linux,
    // and to PyPI (with MPS support) on Apple Silicon.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -e .",
          // Extra dependency for reading .pdf book files in the Audiobook tab
          "uv pip install pypdf"
        ]
      }
    },
    // Linux + AMD needs the ROCm build of torch instead of the CUDA build installed above
    {
      when: "{{platform === 'linux' && gpu === 'amd'}}",
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    // Pre-download the model weights so the first launch only has to load them
    {
      method: "hf.download",
      params: {
        path: "app",
        "_": ["k2-fsa/OmniVoice"]
      }
    },
    // ASR model used to auto-transcribe the reference audio for voice cloning
    {
      method: "hf.download",
      params: {
        path: "app",
        "_": ["openai/whisper-large-v3-turbo"]
      }
    },
    {
      method: "input",
      params: {
        title: "Install Complete!!",
        description: "OmniVoice is installed. Click Start to launch the web UI."
      }
    }
  ]
}
