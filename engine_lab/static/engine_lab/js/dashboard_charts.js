/* Renderizador nativo em Canvas HTML5 para o Protótipo de Dashboard Long & Short */
(function () {
    "use strict";

    function setupCanvas(canvas) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(rect.width, 320);
        const height = Math.max(rect.height, 200);
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);
        return { ctx, width, height };
    }

    function drawHorizontalLine(ctx, y, left, right, color, dashed, label) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        if (dashed) {
            ctx.setLineDash([5, 4]);
        }
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        if (label) {
            ctx.fillStyle = color;
            ctx.font = "10px monospace";
            ctx.textAlign = "right";
            ctx.fillText(label, left - 6, y + 3);
        }
        ctx.restore();
    }

    function drawSeriesLine(ctx, values, mapX, mapY, color, lineWidth) {
        if (!values || values.length === 0) return;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth || 1.8;
        ctx.beginPath();
        for (let i = 0; i < values.length; i++) {
            const x = mapX(i);
            const y = mapY(values[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
    }

    function renderZScoreChart(canvas, payload) {
        if (!canvas || !payload || !payload.z_scores) return;
        const { ctx, width, height } = setupCanvas(canvas);
        const z = payload.z_scores;
        const n = z.length;
        const pad = { top: 18, right: 20, bottom: 26, left: 48 };
        const plotW = width - pad.left - pad.right;
        const plotH = height - pad.top - pad.bottom;

        const minVal = Math.min(-3.4, ...z);
        const maxVal = Math.max(3.4, ...z);

        const mapX = (i) => pad.left + (i / Math.max(1, n - 1)) * plotW;
        const mapY = (v) => pad.top + ((maxVal - v) / (maxVal - minVal)) * plotH;

        ctx.clearRect(0, 0, width, height);

        // Bandas de referência (-3, -2, 0, +2, +3 DP)
        drawHorizontalLine(ctx, mapY(3.0), pad.left, width - pad.right, "rgba(244,63,94,0.45)", true, "+3,0σ");
        drawHorizontalLine(ctx, mapY(2.0), pad.left, width - pad.right, "rgba(245,158,11,0.7)", true, "+2,0σ");
        drawHorizontalLine(ctx, mapY(0.0), pad.left, width - pad.right, "rgba(148,163,184,0.5)", false, "0,0σ");
        drawHorizontalLine(ctx, mapY(-2.0), pad.left, width - pad.right, "rgba(16,185,129,0.7)", true, "-2,0σ");
        drawHorizontalLine(ctx, mapY(-3.0), pad.left, width - pad.right, "rgba(244,63,94,0.45)", true, "-3,0σ");

        // Curva Z-Score
        drawSeriesLine(ctx, z, mapX, mapY, "#38bdf8", 2.0);

        // Destaque no último ponto (Z atual)
        const lastIdx = n - 1;
        const lastX = mapX(lastIdx);
        const lastY = mapY(z[lastIdx]);
        ctx.save();
        ctx.fillStyle = Math.abs(z[lastIdx]) >= 2.0 ? "#10b981" : "#38bdf8";
        ctx.beginPath();
        ctx.arc(lastX, lastY, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Rótulos de datas no eixo X
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px monospace";
        ctx.textAlign = "left";
        ctx.fillText(payload.dates[0], pad.left, height - 8);
        ctx.textAlign = "center";
        ctx.fillText(payload.dates[Math.floor(n / 2)], pad.left + plotW / 2, height - 8);
        ctx.textAlign = "right";
        ctx.fillText(payload.dates[n - 1], width - pad.right, height - 8);
    }

    function renderPricesChart(canvas, payload) {
        if (!canvas || !payload || !payload.prices_a) return;
        const { ctx, width, height } = setupCanvas(canvas);
        const yArr = payload.prices_a;
        const yHat = payload.fitted_values;
        const xArr = payload.prices_b;
        const n = yArr.length;
        const pad = { top: 18, right: 20, bottom: 26, left: 52 };
        const plotW = width - pad.left - pad.right;
        const plotH = height - pad.top - pad.bottom;

        const allVals = yArr.concat(yHat, xArr);
        const minVal = Math.min(...allVals) * 0.985;
        const maxVal = Math.max(...allVals) * 1.015;

        const mapX = (i) => pad.left + (i / Math.max(1, n - 1)) * plotW;
        const mapY = (v) => pad.top + ((maxVal - v) / (maxVal - minVal)) * plotH;

        ctx.clearRect(0, 0, width, height);

        for (let g = 0; g <= 4; g++) {
            const val = minVal + (g / 4) * (maxVal - minVal);
            drawHorizontalLine(
                ctx,
                mapY(val),
                pad.left,
                width - pad.right,
                "rgba(38,53,84,0.65)",
                true,
                val.toFixed(1)
            );
        }

        drawSeriesLine(ctx, xArr, mapX, mapY, "#94a3b8", 1.4);
        drawSeriesLine(ctx, yHat, mapX, mapY, "#f59e0b", 1.6);
        drawSeriesLine(ctx, yArr, mapX, mapY, "#38bdf8", 2.0);

        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px monospace";
        ctx.textAlign = "left";
        ctx.fillText(payload.dates[0], pad.left, height - 8);
        ctx.textAlign = "right";
        ctx.fillText(payload.dates[n - 1], width - pad.right, height - 8);
    }

    function renderBetaRotationChart(canvas, payload) {
        if (!canvas || !payload || !payload.beta_rotation || payload.beta_rotation.length === 0) return;
        const { ctx, width, height } = setupCanvas(canvas);
        const b = payload.beta_rotation;
        const n = b.length;
        const pad = { top: 18, right: 20, bottom: 26, left: 52 };
        const plotW = width - pad.left - pad.right;
        const plotH = height - pad.top - pad.bottom;

        const upper = payload.beta_rotation_upper;
        const lower = payload.beta_rotation_lower;
        const mean = payload.beta_rotation_mean;

        const minVal = Math.min(lower, ...b) - 0.02;
        const maxVal = Math.max(upper, ...b) + 0.02;

        const mapX = (i) => pad.left + (i / Math.max(1, n - 1)) * plotW;
        const mapY = (v) => pad.top + ((maxVal - v) / (maxVal - minVal)) * plotH;

        ctx.clearRect(0, 0, width, height);

        drawHorizontalLine(ctx, mapY(upper), pad.left, width - pad.right, "rgba(244,63,94,0.6)", true, upper.toFixed(3));
        drawHorizontalLine(ctx, mapY(mean), pad.left, width - pad.right, "rgba(148,163,184,0.6)", false, mean.toFixed(3));
        drawHorizontalLine(ctx, mapY(lower), pad.left, width - pad.right, "rgba(244,63,94,0.6)", true, lower.toFixed(3));

        drawSeriesLine(ctx, b, mapX, mapY, "#10b981", 2.0);

        const dates = payload.beta_rotation_dates || [];
        if (dates.length > 0) {
            ctx.fillStyle = "#94a3b8";
            ctx.font = "10px monospace";
            ctx.textAlign = "left";
            ctx.fillText(dates[0], pad.left, height - 8);
            ctx.textAlign = "right";
            ctx.fillText(dates[dates.length - 1], width - pad.right, height - 8);
        }
    }

    function initDashboardCharts() {
        const scriptEl = document.getElementById("dashboard-chart-payload");
        if (!scriptEl) return;
        let payload = null;
        try {
            payload = JSON.parse(scriptEl.textContent);
        } catch (err) {
            return;
        }

        const renderAll = () => {
            renderZScoreChart(document.getElementById("zscore-canvas"), payload);
            renderPricesChart(document.getElementById("prices-canvas"), payload);
            renderBetaRotationChart(document.getElementById("beta-rotation-canvas"), payload);
        };

        renderAll();
        window.addEventListener("resize", renderAll);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initDashboardCharts);
    } else {
        initDashboardCharts();
    }
})();
