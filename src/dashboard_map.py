from __future__ import annotations

import panel as pn


TOPO_ANIMATION_JS = """
<script>
(function animate() {
  if (window.deck) {
    const layers = window.deck.props.layers;
    if (layers) {
      let updated = false;
      const newLayers = layers.map(layer => {
        if (layer.id === 'topology_trips_layer') {
          updated = true;
          return layer.clone({ current_time: (layer.props.current_time + 1) % 100 });
        }
        return layer;
      });
      if (updated) {
        window.deck.setProps({ layers: newLayers });
      }
    }
  }
  requestAnimationFrame(animate);
})();
</script>
"""


def topo_animation_pane() -> pn.pane.HTML:
    """Client-side animation loop for TripsLayer current_time."""
    return pn.pane.HTML(TOPO_ANIMATION_JS, width=0, height=0, margin=0, sizing_mode="fixed")


def compose_map_panel(deck_panel, campus_warning=None):
    """Compose warning + deck panel in a stable layout."""
    if campus_warning is not None:
        return pn.Column(campus_warning, deck_panel, sizing_mode="stretch_width")
    return deck_panel
