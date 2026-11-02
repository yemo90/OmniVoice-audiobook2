module.exports = {
  run: [
    // Pull the launcher itself (only when it lives in a git repository)
    {
      when: "{{exists('.git')}}",
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    // Pick up any new dependencies from the updated source
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -e .",
          "uv pip install pypdf"
        ]
      }
    },
    // Refresh the model weights (no-op when they are already up to date)
    {
      method: "hf.download",
      params: {
        path: "app",
        "_": ["k2-fsa/OmniVoice"]
      }
    },
    {
      method: "hf.download",
      params: {
        path: "app",
        "_": ["openai/whisper-large-v3-turbo"]
      }
    }
  ]
}
