module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          "PYTORCH_ENABLE_MPS_FALLBACK": 1,
          // Keep stdout line-buffered so the "Running on local URL" line below
          // reaches Pinokio as soon as Gradio prints it, not when a pipe fills.
          "PYTHONUNBUFFERED": 1
        },
        path: "app",
        message: [
          // OmniVoice Studio: upstream demo UI + a third "Audiobook" tab
          "python -u ../audiobook/studio.py --ip 127.0.0.1 --port {{port}}"
        ],
        on: [{
          // Gradio prints "Running on local URL: http://127.0.0.1:<port>"
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      // This step sets the local variable 'url'.
      // This local variable will be used in pinokio.js to display the "Open WebUI" tab when the value is set.
      method: "local.set",
      params: {
        // the input.event is the regular expression match object from the previous step
        url: "{{input.event[1]}}"
      }
    }
  ]
}
