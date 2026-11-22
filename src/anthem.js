(() => {
    // Keep one player alive across Streamlit updates and tab switches.
    if (!window.apexAnthem) {
        const player = document.createElement("audio");
        player.id = "apex-background-anthem";
        player.loop = true;
        player.preload = "auto";
        player.volume = 0.3;
        player.hidden = true;
        document.body.append(player);

        let panel;
        let automatic = true;
        let state = "loading";
        let attempt = 0;

        function update() {
            if (!panel?.isConnected || !panel.dataset.source) return;
            const button = panel.querySelector("[data-anthem-toggle]");
            button.textContent = automatic && (state === "playing" || state === "loading") ? "Jeda" : "Putar";
            button.setAttribute("aria-label", `${button.textContent} lagu kebangsaan`);
            panel.querySelector("[data-anthem-status]").textContent = {
                playing: "Diputar berulang",
                loading: "Memuat audio…",
                blocked: "Mulai setelah Anda berinteraksi",
                paused: "Dijeda",
                error: "Audio gagal dimuat",
            }[state];
        }

        async function start() {
            if (!automatic || !player.hasAttribute("src")) return;
            const request = ++attempt;
            state = "loading";
            update();
            try {
                await player.play();
                if (request === attempt) state = "playing";
            } catch (error) {
                if (request !== attempt) return;
                state = error.name === "NotAllowedError" ? "blocked" : "error";
            }
            if (request === attempt) update();
        }

        function resumeAfterInteraction(event) {
            if (!event.isTrusted || state !== "blocked" || event.target.closest?.("[data-anthem-toggle]")) return;
            start();
        }
        document.addEventListener("click", resumeAfterInteraction, { capture: true });
        document.addEventListener("keydown", resumeAfterInteraction, { capture: true });

        window.apexAnthem = (nextPanel) => {
            panel = nextPanel;
            const source = panel?.dataset.source;
            if (!source) {
                ++attempt;
                player.pause();
                player.removeAttribute("src");
                player.load();
                return;
            }
            panel.querySelector("[data-anthem-toggle]").onclick = () => {
                if (automatic && (state === "playing" || state === "loading")) {
                    automatic = false;
                    ++attempt;
                    player.pause();
                    state = "paused";
                    update();
                } else {
                    automatic = true;
                    start();
                }
            };
            if (player.getAttribute("src") === source) {
                update();
                return;
            }
            ++attempt;
            player.pause();
            player.src = source;
            if (automatic) {
                start();
            } else {
                state = "paused";
                update();
            }
        };
    }

    window.apexAnthem(document.querySelector("[data-apex-anthem]"));
})();
