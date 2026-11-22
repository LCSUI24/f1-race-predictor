(() => {
    const map = document.querySelector("[data-apex-gap-map]");
    if (!map || map.dataset.ready) return;

    function shape(tag, attrs = {}) {
        const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const [name, value] of Object.entries(attrs)) element.setAttribute(name, value);
        return element;
    }

    const diagram = map.querySelector("[data-track-diagram]");
    const fallback = diagram.querySelector("img");
    const svg = shape("svg", {
        viewBox: "-25 -25 550 550",
        role: "img",
        "aria-label": fallback.alt,
    });
    const edge = shape("path", { class: "track-edge", d: diagram.dataset.path });
    const path = shape("path", { class: "track-line", d: diagram.dataset.path });
    const finishMark = shape("g", { "data-track-finish": "" });
    finishMark.append(shape("line", { x1: 0, x2: 0, y1: -15, y2: 15 }));
    const markerLayer = shape("g");
    const label = shape("text", { class: "track-marker-label" });
    svg.append(edge, path, finishMark, markerLayer, label);
    diagram.prepend(svg);

    const length = path.getTotalLength();
    if (!length) {
        svg.remove();
        return;
    }
    fallback.hidden = true;

    const wrap = (fraction) => ((fraction % 1) + 1) % 1;
    const at = (fraction) => path.getPointAtLength(length * wrap(fraction));
    const start = Number(map.dataset.start);
    const direction = Number(map.dataset.direction);
    const lapScale = Number(map.dataset.lapScale);
    const finish = at(start);
    const next = at(start + direction * 0.008);
    const heading = Math.atan2(next.y - finish.y, next.x - finish.x) * 180 / Math.PI;
    finishMark.setAttribute("transform", `translate(${finish.x} ${finish.y}) rotate(${heading})`);

    const rows = Array.from(map.querySelectorAll("[data-driver-row]"));
    const points = rows.map((button) => {
        const { driverRow, code, seconds, tint } = button.dataset;
        const fraction = start - direction * Number(seconds) / lapScale;
        const { x, y } = at(fraction);
        const dot = shape("circle", { "data-driver-dot": driverRow, cx: x, cy: y, r: 6.5, fill: tint });
        const title = shape("title");
        title.textContent = Array.from(button.querySelectorAll(".gap-pos, strong, .gap-value"))
            .map((part) => part.textContent).join(" ");
        dot.append(title);
        return { button, dot, x, y, code };
    });

    function highlight(point) {
        for (const entry of points) {
            entry.button.setAttribute("aria-pressed", String(entry === point));
            entry.dot.classList.toggle("is-selected", entry === point);
        }
        markerLayer.append(point.dot);
        const alignRight = point.x > 400;
        label.setAttribute("x", point.x + (alignRight ? -13 : 13));
        label.setAttribute("y", point.y + (point.y < 25 ? 25 : -12));
        label.setAttribute("text-anchor", alignRight ? "end" : "start");
        label.textContent = point.code;
    }

    for (const point of points.slice().reverse()) {
        markerLayer.append(point.dot);
        point.button.addEventListener("click", () => highlight(point));
        point.button.addEventListener("focus", () => highlight(point));
        point.dot.addEventListener("click", () => highlight(point));
    }

    map.dataset.ready = "true";
    if (points.length) highlight(points[0]);
})();
