import panel as pn
import pandas as pd
from typing import Dict, Any

pn.extension('echarts')

class TopologyView(pn.viewable.Viewer):
    def __init__(self, topo_manager, current_power_df, **params):
        super().__init__(**params)
        self.topo_manager = topo_manager
        self.current_power_df = current_power_df # Used to calculate current load values if needed
        # We will render an ECharts Tree
        self._echart_pane = pn.pane.ECharts(self._get_echarts_dict(), height=600, sizing_mode="stretch_width")

    def _build_echart_tree(self, node) -> Dict[str, Any]:
        """Convert GridNode to ECharts series data format recursively."""
        # Calculate current power for this node if we want to show it
        # For simplicity, we can do it here but it might be heavy. Let's just show structure first.
        value = 0
        if len(self.current_power_df) > 0:
            agg_series = self.topo_manager.aggregate_power(self.current_power_df, node.node_id)
            if not agg_series.empty:
                value = round(agg_series.iloc[-1], 2) # Use the latest hour as current load
        
        name_label = f"{node.node_id}\n({value} kW)" if value > 0 else node.node_id
        if node.level == 'METER' and 'name' in node.metadata:
            name_label = f"{node.node_id}\n{node.metadata['name']}"
            
        data = {
            "name": name_label,
            "value": value,
        }
        
        if node.children:
            data["children"] = [self._build_echart_tree(c) for c in node.children]
            
        return data

    def _get_echarts_dict(self):
        tree_data = [self._build_echart_tree(self.topo_manager.root)]
        
        return {
            "tooltip": {
                "trigger": "item",
                "triggerOn": "mousemove"
            },
            "series": [
                {
                    "type": "tree",
                    "data": tree_data,
                    "top": "1%",
                    "left": "7%",
                    "bottom": "1%",
                    "right": "20%",
                    "symbolSize": 7,
                    "label": {
                        "position": "left",
                        "verticalAlign": "middle",
                        "align": "right",
                        "fontSize": 12,
                        "color": "#fff" # Dark theme friendly
                    },
                    "leaves": {
                        "label": {
                            "position": "right",
                            "verticalAlign": "middle",
                            "align": "left",
                        }
                    },
                    "emphasis": {
                        "focus": "descendant"
                    },
                    "expandAndCollapse": True,
                    "animationDuration": 550,
                    "animationDurationUpdate": 750
                }
            ]
        }

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("### 電網拓樸樹狀圖 (Power Grid Topology)"),
            self._echart_pane,
            sizing_mode="stretch_both"
        )
