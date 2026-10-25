
function d100Visualizer(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const base =
    '../figures/chapter05/covariance_structure/d100/';

  const warmupValues = [10, 20, 30, 40, 50, 100, 200, 500, 1000];

  const slider = container.querySelector('#d100-w-slider');
  const valueLabel = container.querySelector('#d100-w-value');
  const plot = container.querySelector('#d100-plot');
  const title = container.querySelector('#d100-plot-title');
  const status = container.querySelector('#d100-plot-status');
  const buttons = Array.from(
    container.querySelectorAll('.d100-btn')
  );

  if (!slider || !plot || buttons.length === 0) return;

  let selectedType = 'diagonal';

  slider.min = 0;
  slider.max = warmupValues.length - 1;
  slider.step = 1;

  function getPlotFilename(type, W) {
    const prefix = type === 'diagonal' ? 'D' : 'T';
    return `highlight_${prefix}_W${W}.png`;
    }

  function updatePlot() {
    const W = warmupValues[Number(slider.value)];

    const typeLabel = selectedType === 'diagonal'
      ? 'Diagonal'
      : 'Transformed';

    plot.src = base + getPlotFilename(selectedType, W);
    plot.alt =
      `100-dimensional MVN ${typeLabel.toLowerCase()} plot at W = ${W}`;

    valueLabel.textContent = W;
    title.textContent = `${typeLabel} — W = ${W}`;
    status.textContent = `W = ${W}`;

    buttons.forEach(button => {
      const active = button.dataset.image === selectedType;

      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
  }

  slider.addEventListener('input', updatePlot);

  buttons.forEach(button => {
    button.addEventListener('click', () => {
      selectedType = button.dataset.image;
      updatePlot();
    });
  });

  updatePlot();
}
