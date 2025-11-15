import React, { useState } from "react";
import CameraRecorder from "../components/CameraRecorder";
import Timeline from "../components/Timeline";

function Recording() {
  const [isRecording, setIsRecording] = useState(false);
  const [events, setEvents] = useState([]);

  const handleStartRecording = () => {
    setIsRecording(true);
  };

  const handleStopRecording = () => {
    setIsRecording(false);
  };

  const handleRecordingStart = () => {
    console.log("Recording started");
  };

  const handleRecordingStop = (blob) => {
    console.log("Recording stopped", blob);
  };

  return (
    <main className="flex-1 bg-primary-50 flex items-center justify-center p-8">
      <div className="flex w-full max-w-7xl gap-8">
        <div className="flex-1 flex flex-col items-center justify-center">
          <CameraRecorder
            onRecordingStart={handleRecordingStart}
            onRecordingStop={handleRecordingStop}
            isRecording={isRecording}
          />
          <div className="mt-6 flex space-x-4">
            <button
              onClick={handleStartRecording}
              disabled={isRecording}
              className={`px-6 py-3 rounded-lg font-medium transition-colors duration-200 shadow-sm ${
                isRecording
                  ? "bg-gray-400 text-gray-200 cursor-not-allowed"
                  : "bg-primary-600 text-white hover:bg-primary-700"
              }`}
            >
              Start Recording
            </button>
            <button
              onClick={handleStopRecording}
              disabled={!isRecording}
              className={`px-6 py-3 rounded-lg font-medium transition-colors duration-200 shadow-sm ${
                !isRecording
                  ? "bg-gray-400 text-gray-200 cursor-not-allowed"
                  : "bg-red-600 text-white hover:bg-red-700"
              }`}
            >
              Stop Recording
            </button>
          </div>
        </div>
        <div className="w-24 flex flex-col items-center">
          <div className="h-full min-h-[600px]">
            <Timeline events={events} />
          </div>
        </div>
      </div>
    </main>
  );
}

export default Recording;

