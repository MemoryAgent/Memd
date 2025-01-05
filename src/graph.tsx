import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import * as graph_data_example from "@/data/example.json";

const ChartOne = () => {
    const chartRef = useRef<HTMLDivElement>(null);
    const exampleData = graph_data_example;

    useEffect(() => {
        // Initialize the chart only once
        let chartInstance: echarts.EChartsType | null = null;

        if (chartRef.current) {
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
                        layout: "none",
                        data: exampleData.nodes.map((node) => ({
                            id: node.id,
                            name: node.summary ?? node.gist,
                            symbolSize: 26.6666666666666665,
                            x: node.x,
                            y: node.y,
                            value: 30,
                            category: 1,
                        })),
                        links: exampleData.link.map((link) => ({
                            source: link.from,
                            target: link.to,
                        })),
                        roam: true,
                        label: {
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
        }

        // Cleanup function to dispose of the chart instance
        return () => {
            chartInstance?.dispose();
        };
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
