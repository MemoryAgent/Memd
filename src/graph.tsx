import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import graph_data_example from "@/data/example.ts";
import { MemoryNode } from "./data/node";

const ChartOne = () => {
    const chartRef = useRef<HTMLDivElement>(null);
    const exampleData = graph_data_example;

    useEffect(() => {
        let chartInstance: echarts.EChartsType | null = null;

        if (chartRef.current) {
            // Initialize the chart
            chartInstance = echarts.init(chartRef.current);

            const option: echarts.EChartsOption = {
                title: {
                    text: "Memory",
                    subtext: "Default layout",
                    top: "bottom",
                    left: "right",
                },
                tooltip: {},
                legend: [],
                animationDuration: 1500,
                animationEasingUpdate: "quinticInOut",
                series: [
                    {
                        name: "Memory",
                        type: "graph",
                        legendHoverLink: false,
                        layout: "force",
                        data: exampleData.nodes.map((node: MemoryNode) => ({
                            id: node.id,
                            name:
                                node.kind === "leaf" ? node.gist : node.summary,
                            symbolSize: node.kind == "leaf" ? 10 : 20,
                            value: 30,
                            category: node.kind == "leaf" ? 1 : 2,
                        })),
                        links: exampleData.link.map((link) => ({
                            source: link.from,
                            target: link.to,
                        })),
                        roam: true,
                        force: {
                            repulsion: 100,
                        },
                        label: {
                            show: true,
                            position: "right",
                            formatter: "{b}",
                        },
                        lineStyle: {
                            color: "source",
                            curveness: 0.1,
                        },
                        emphasis: {
                            focus: "adjacency",
                            lineStyle: {
                                width: 10,
                            },
                        },
                    },
                ],
            };

            chartInstance.setOption(option);

            // Add a resize event listener
            const handleResize = () => {
                chartInstance?.resize();
            };

            window.addEventListener("resize", handleResize);

            // Cleanup function
            return () => {
                chartInstance?.dispose();
                window.removeEventListener("resize", handleResize);
            };
        }
    }, [exampleData]);

    return <div ref={chartRef} className="flex w-full min-h-screen"></div>;
};

export default function Graph() {
    return (
        <div className="flex flex-col items-center justify-center h-full w-full">
            <ChartOne />
        </div>
    );
}
