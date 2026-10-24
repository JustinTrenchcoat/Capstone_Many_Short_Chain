// ============================================================
// Interactive Bivariate Normal Distribution Visualizer
// ============================================================

function bvnVisualizer(containerId) {

  const container = document.getElementById(containerId);

  if (!container) {
    console.error(`Container #${containerId} not found.`);
    return;
  }

  // ----------------------------------------------------------
  // Configuration
  // ----------------------------------------------------------

  const width = 700;
  const height = 650;

  const margin = {
    top: 30,
    right: 30,
    bottom: 60,
    left: 70
  };

  // ----------------------------------------------------------
  // HTML
  // ----------------------------------------------------------

  container.innerHTML = `
    <div class="bvn-wrapper">

      <div class="bvn-controls">

        <div class="bvn-distribution">
          <h3>Target Distribution</h3>

          <div class="bvn-row">
            <label>
              Mean X
              <input id="${containerId}-m1x"
                     type="number"
                     value="0"
                     step="0.1">
            </label>

            <label>
              Mean Y
              <input id="${containerId}-m1y"
                     type="number"
                     value="0"
                     step="0.1">
            </label>
          </div>

          <div class="bvn-row">
            <label>
              Variance X
              <input id="${containerId}-v1x"
                     type="number"
                     value="1"
                     min="0.001"
                     step="0.1">
            </label>

            <label>
              Variance Y
              <input id="${containerId}-v1y"
                     type="number"
                     value="1"
                     min="0.001"
                     step="0.1">
            </label>
          </div>

          <label>
            Correlation ρ
            <input id="${containerId}-r1"
                   type="number"
                   value="0"
                   min="-0.99"
                   max="0.99"
                   step="0.05">
          </label>

        </div>


        <div class="bvn-distribution">

          <h3>Initialization Distribution</h3>

          <div class="bvn-row">

            <label>
              Mean X
              <input id="${containerId}-m2x"
                     type="number"
                     value="1"
                     step="0.1">
            </label>

            <label>
              Mean Y
              <input id="${containerId}-m2y"
                     type="number"
                     value="0.5"
                     step="0.1">
            </label>

          </div>

          <div class="bvn-row">

            <label>
              Variance X
              <input id="${containerId}-v2x"
                     type="number"
                     value="2"
                     min="0.001"
                     step="0.1">
            </label>

            <label>
              Variance Y
              <input id="${containerId}-v2y"
                     type="number"
                     value="1"
                     min="0.001"
                     step="0.1">
            </label>

          </div>

          <label>
            Correlation ρ
            <input id="${containerId}-r2"
                   type="number"
                   value="0.6"
                   min="-0.99"
                   max="0.99"
                   step="0.05">
          </label>

        </div>

      </div>


      <div class="bvn-plot-container">

        <svg
          id="${containerId}-svg"
          viewBox="0 0 ${width} ${height}"
          preserveAspectRatio="xMidYMid meet">
        </svg>

        <div class="bvn-legend">

          <span>
            <span class="bvn-line dist1"></span>
            Target Distribution — 95%
          </span>

          <span>
            <span class="bvn-line dashed dist1"></span>
            Target Distribution — 68%
          </span>

          <span>
            <span class="bvn-line dist2"></span>
            Initialization Distribution — 95%
          </span>

          <span>
            <span class="bvn-line dashed dist2"></span>
            Initialization Distribution — 68%
          </span>

        </div>

      </div>

    </div>
  `;


  // ----------------------------------------------------------
  // CSS
  // ----------------------------------------------------------

  const style = document.createElement("style");

  style.textContent = `

    .bvn-wrapper {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 25px;
      margin: 20px 0;
      align-items: start;
    }

    .bvn-controls {
      display: flex;
      flex-direction: column;
      gap: 15px;
    }

    .bvn-distribution {
      border: 1px solid #cccccc;
      border-radius: 8px;
      padding: 15px;
    }

    .bvn-distribution h3 {
      margin-top: 0;
      font-size: 1rem;
    }

    .bvn-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .bvn-distribution label {
      display: block;
      font-size: 0.85rem;
      margin-bottom: 10px;
    }

    .bvn-distribution input {
      width: 100%;
      box-sizing: border-box;
      margin-top: 4px;
      padding: 6px;
      border: 1px solid #aaa;
      border-radius: 5px;
    }

    .bvn-plot-container {
      min-width: 0;
    }

    .bvn-plot-container svg {
      width: 100%;
      max-width: ${width}px;
      height: auto;
      display: block;
      border: 1px solid #cccccc;
      border-radius: 8px;
    }

    .bvn-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 15px;
      margin-top: 10px;
      font-size: 0.8rem;
    }

    .bvn-line {
      display: inline-block;
      width: 25px;
      margin-right: 5px;
      vertical-align: middle;
      border-top: 2px solid;
    }

    .bvn-line.dashed {
      border-top-style: dashed;
    }

    .bvn-line.dist1 {
      color: #2166ac;
    }

    .bvn-line.dist2 {
      color: #b2182b;
    }

    @media (max-width: 750px) {

      .bvn-wrapper {
        grid-template-columns: 1fr;
      }

    }

  `;

  container.appendChild(style);


  // ----------------------------------------------------------
  // Helper functions
  // ----------------------------------------------------------

  function getValue(id) {

    const element =
      document.getElementById(`${containerId}-${id}`);

    return Number(element.value);

  }


  // ----------------------------------------------------------
  // Chi-square radius
  //
  // For df = 2:
  //
  // chi2.ppf(p, 2) = -2 log(1-p)
  //
  // ----------------------------------------------------------

  function chiRadius(probability) {

    return Math.sqrt(
      -2 * Math.log(1 - probability)
    );

  }


  // ----------------------------------------------------------
  // Generate ellipse
  // ----------------------------------------------------------

  function generateEllipse(
    meanX,
    meanY,
    varX,
    varY,
    rho,
    probability
  ) {

    // Covariance

    const covariance =
      rho * Math.sqrt(varX * varY);


    const a = varX;
    const b = covariance;
    const d = varY;


    // Eigenvalues

    const trace = a + d;

    const discriminant =
      Math.sqrt(
        (a - d) ** 2 +
        4 * b ** 2
      );


    const lambda1 =
      (trace + discriminant) / 2;

    const lambda2 =
      (trace - discriminant) / 2;


    // Eigenvector orientation

    let angle = 0;

    if (Math.abs(b) > 1e-12) {

      angle =
        Math.atan2(
          lambda1 - a,
          b
        );

    } else if (a < d) {

      angle = Math.PI / 2;

    }


    // Mahalanobis radius

    const radius =
      chiRadius(probability);


    const points = [];

    const n = 300;


    for (let i = 0; i <= n; i++) {

      const theta =
        2 * Math.PI * i / n;


      // Circle

      const x0 =
        radius *
        Math.sqrt(lambda1) *
        Math.cos(theta);

      const y0 =
        radius *
        Math.sqrt(lambda2) *
        Math.sin(theta);


      // Rotate

      const x =
        meanX +
        x0 * Math.cos(angle) -
        y0 * Math.sin(angle);

      const y =
        meanY +
        x0 * Math.sin(angle) +
        y0 * Math.cos(angle);


      points.push([x, y]);

    }


    return points;

  }


  // ----------------------------------------------------------
  // Draw
  // ----------------------------------------------------------

  function draw() {

    const svg =
      document.getElementById(
        `${containerId}-svg`
      );


    // --------------------------------------------------------
    // Read parameters
    // --------------------------------------------------------

    const d1 = {

      meanX: getValue("m1x"),
      meanY: getValue("m1y"),

      varX: Math.max(
        0.001,
        getValue("v1x")
      ),

      varY: Math.max(
        0.001,
        getValue("v1y")
      ),

      rho: Math.max(
        -0.99,
        Math.min(
          0.99,
          getValue("r1")
        )
      )

    };


    const d2 = {

      meanX: getValue("m2x"),
      meanY: getValue("m2y"),

      varX: Math.max(
        0.001,
        getValue("v2x")
      ),

      varY: Math.max(
        0.001,
        getValue("v2y")
      ),

      rho: Math.max(
        -0.99,
        Math.min(
          0.99,
          getValue("r2")
        )
      )

    };


    // --------------------------------------------------------
    // Generate ellipses
    // --------------------------------------------------------

    const ellipses = [

      {
        distribution: d1,
        probability: 0.95,
        dashed: false,
        className: "dist1"
      },

      {
        distribution: d1,
        probability: 0.68,
        dashed: true,
        className: "dist1"
      },

      {
        distribution: d2,
        probability: 0.95,
        dashed: false,
        className: "dist2"
      },

      {
        distribution: d2,
        probability: 0.68,
        dashed: true,
        className: "dist2"
      }

    ];


    // --------------------------------------------------------
    // Determine plot limits
    // --------------------------------------------------------

    const allPoints = [];

    ellipses.forEach(e => {

      const points =
        generateEllipse(
          e.distribution.meanX,
          e.distribution.meanY,
          e.distribution.varX,
          e.distribution.varY,
          e.distribution.rho,
          e.probability
        );

      allPoints.push(...points);

    });


    let xmin =
      Math.min(...allPoints.map(p => p[0]));

    let xmax =
      Math.max(...allPoints.map(p => p[0]));

    let ymin =
      Math.min(...allPoints.map(p => p[1]));

    let ymax =
      Math.max(...allPoints.map(p => p[1]));


    const xPadding =
      (xmax - xmin) * 0.12 || 1;

    const yPadding =
      (ymax - ymin) * 0.12 || 1;


    xmin -= xPadding;
    xmax += xPadding;

    ymin -= yPadding;
    ymax += yPadding;


    // --------------------------------------------------------
    // Coordinate transformation
    // --------------------------------------------------------

    const plotWidth =
      width -
      margin.left -
      margin.right;

    const plotHeight =
      height -
      margin.top -
      margin.bottom;


    function sx(x) {

      return (
        margin.left +
        (x - xmin) /
        (xmax - xmin) *
        plotWidth
      );

    }


    function sy(y) {

      return (
        height -
        margin.bottom -
        (y - ymin) /
        (ymax - ymin) *
        plotHeight
      );

    }


    // --------------------------------------------------------
    // Clear SVG
    // --------------------------------------------------------

    svg.innerHTML = "";


    // --------------------------------------------------------
    // SVG helpers
    // --------------------------------------------------------

    function createElement(
      type,
      attributes = {}
    ) {

      const element =
        document.createElementNS(
          "http://www.w3.org/2000/svg",
          type
        );

      Object.entries(attributes).forEach(
        ([key, value]) => {

          element.setAttribute(
            key,
            value
          );

        }
      );

      return element;

    }


    // --------------------------------------------------------
    // Grid
    // --------------------------------------------------------

    const gridGroup =
      createElement("g");


    const numberOfTicks = 8;


    for (
      let i = 0;
      i <= numberOfTicks;
      i++
    ) {

      const x =
        xmin +
        (xmax - xmin) *
        i / numberOfTicks;

      const line =
        createElement(
          "line",
          {
            x1: sx(x),
            x2: sx(x),
            y1: margin.top,
            y2: height - margin.bottom,
            stroke: "currentColor",
            opacity: "0.08"
          }
        );

      gridGroup.appendChild(line);


      const text =
        createElement(
          "text",
          {
            x: sx(x),
            y: height - margin.bottom + 20,
            "text-anchor": "middle",
            "font-size": "11",
            fill: "currentColor"
          }
        );

      text.textContent =
        x.toFixed(1);

      gridGroup.appendChild(text);

    }


    for (
      let i = 0;
      i <= numberOfTicks;
      i++
    ) {

      const y =
        ymin +
        (ymax - ymin) *
        i / numberOfTicks;


      const line =
        createElement(
          "line",
          {
            x1: margin.left,
            x2: width - margin.right,
            y1: sy(y),
            y2: sy(y),
            stroke: "currentColor",
            opacity: "0.08"
          }
        );

      gridGroup.appendChild(line);


      const text =
        createElement(
          "text",
          {
            x: margin.left - 8,
            y: sy(y) + 4,
            "text-anchor": "end",
            "font-size": "11",
            fill: "currentColor"
          }
        );

      text.textContent =
        y.toFixed(1);

      gridGroup.appendChild(text);

    }


    svg.appendChild(gridGroup);


    // --------------------------------------------------------
    // Axes
    // --------------------------------------------------------

    const xAxis =
      createElement(
        "line",
        {
          x1: margin.left,
          x2: width - margin.right,
          y1: sy(0),
          y2: sy(0),
          stroke: "currentColor",
          opacity: "0.35"
        }
      );


    const yAxis =
      createElement(
        "line",
        {
          x1: sx(0),
          x2: sx(0),
          y1: margin.top,
          y2: height - margin.bottom,
          stroke: "currentColor",
          opacity: "0.35"
        }
      );


    svg.appendChild(xAxis);
    svg.appendChild(yAxis);


    // --------------------------------------------------------
    // Axis labels
    // --------------------------------------------------------

    const xlabel =
      createElement(
        "text",
        {
          x: (margin.left + width - margin.right) / 2,
          y: height - 15,
          "text-anchor": "middle",
          "font-size": "15",
          fill: "currentColor"
        }
      );

    xlabel.textContent = "X₁";

    svg.appendChild(xlabel);


    const ylabel =
      createElement(
        "text",
        {
          x: 18,
          y: (margin.top + height - margin.bottom) / 2,
          "text-anchor": "middle",
          "font-size": "15",
          fill: "currentColor",
          transform:
            `rotate(-90 18 ${(margin.top + height - margin.bottom) / 2})`
        }
      );

    ylabel.textContent = "X₂";

    svg.appendChild(ylabel);


    // --------------------------------------------------------
    // Draw ellipses
    // --------------------------------------------------------

    ellipses.forEach(e => {

      const points =
        generateEllipse(
          e.distribution.meanX,
          e.distribution.meanY,
          e.distribution.varX,
          e.distribution.varY,
          e.distribution.rho,
          e.probability
        );


      const d =
        points
          .map(
            (p, i) =>
              `${i === 0 ? "M" : "L"} ${sx(p[0])} ${sy(p[1])}`
          )
          .join(" ") +
        " Z";


      const path =
        createElement(
          "path",
          {
            d: d,
            fill: "none",
            stroke:
              e.className === "dist1"
                ? "#2166ac"
                : "#b2182b",
            "stroke-width": "2.5"
          }
        );


      if (e.dashed) {

        path.setAttribute(
          "stroke-dasharray",
          "8 6"
        );

      }


      svg.appendChild(path);

    });


    // --------------------------------------------------------
    // Draw means
    // --------------------------------------------------------

    function drawMean(
      x,
      y,
      color,
      type
    ) {

      if (type === "circle") {

        const circle =
          createElement(
            "circle",
            {
              cx: sx(x),
              cy: sy(y),
              r: 6,
              fill: color
            }
          );

        svg.appendChild(circle);

      } else {

        const line1 =
          createElement(
            "line",
            {
              x1: sx(x) - 6,
              x2: sx(x) + 6,
              y1: sy(y) - 6,
              y2: sy(y) + 6,
              stroke: color,
              "stroke-width": 3
            }
          );


        const line2 =
          createElement(
            "line",
            {
              x1: sx(x) - 6,
              x2: sx(x) + 6,
              y1: sy(y) + 6,
              y2: sy(y) - 6,
              stroke: color,
              "stroke-width": 3
            }
          );


        svg.appendChild(line1);
        svg.appendChild(line2);

      }

    }


    drawMean(
      d1.meanX,
      d1.meanY,
      "#2166ac",
      "circle"
    );


    drawMean(
      d2.meanX,
      d2.meanY,
      "#b2182b",
      "x"
    );

  }


  // ----------------------------------------------------------
  // Connect inputs to plot
  // ----------------------------------------------------------

  const inputIds = [

    "m1x",
    "m1y",
    "v1x",
    "v1y",
    "r1",

    "m2x",
    "m2y",
    "v2x",
    "v2y",
    "r2"

  ];


  inputIds.forEach(id => {

    document
      .getElementById(`${containerId}-${id}`)
      .addEventListener(
        "input",
        draw
      );

  });


  // Initial plot

  draw();

}