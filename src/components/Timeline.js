import React from "react";

function Timeline({ events = [] }) {
  return (
    <div className="flex flex-col items-center h-full">
      <div className="relative h-full w-1 bg-primary-300 rounded-full">
        {events.map((event, index) => (
          <div
            key={index}
            className={`absolute w-4 h-4 rounded border-2 ${
              event.side === "left"
                ? "left-[-8px] border-primary-600 bg-white"
                : "right-[-8px] border-primary-600 bg-white"
            }`}
            style={{
              top: `${(index / Math.max(events.length - 1, 1)) * 100}%`,
            }}
            title={event.label || `Event ${index + 1}`}
          ></div>
        ))}
      </div>
    </div>
  );
}

export default Timeline;


