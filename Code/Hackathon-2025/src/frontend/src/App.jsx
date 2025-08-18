import { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  // State management
  const [jsonData, setJsonData] = useState(null);
  const [editableJson, setEditableJson] = useState("");
  const [status, setStatus] = useState("pending");
  const [micStatus, setMicStatus] = useState("off");
  const [isEditing, setIsEditing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  // Refs for audio recording
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Audio recording functions
  const toggleRecording = async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  };

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav",
        });
        await processRecording(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setMicStatus("on");
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setError("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream
        .getTracks()
        .forEach((track) => track.stop());
      setIsRecording(false);
      setMicStatus("off");
    }
  };

  const processRecording = async (audioBlob) => {
    try {
      setProcessing(true);
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.wav");

      // Send recording to backend for processing
      const response = await axios.post(
        "http://localhost:8000/save-recording",
        formData,
        { responseType: "json" },
      );
      const serverString = response.data;
      console.log("Server response:", serverString);
      // Set the JSON data from backend response
      setJsonData(response.data);
      //setServerResponse(JSON.parse(jsonData))
      setEditableJson(JSON.stringify(response.data, null, 2));
      setProcessing(false);
    } catch (error) {
      console.error("Error processing recording:", error);
      setError("Failed to process recording. Please try again.");
      setProcessing(false);
    }
  };

  // JSON editing functions
  const handleApprove = () => {
    setStatus("approved");
  };

  const handleEditToggle = () => {
    if (isEditing) {
      try {
        const parsedData = JSON.parse(editableJson);
        setJsonData(parsedData);
        setStatus("edited");

        // Send updated JSON to backend
        axios
          .post("http://localhost:8000/update-json", parsedData)
          .then(() => console.log("Data updated successfully"))
          .catch((err) => console.error("Update failed:", err));
      } catch (error) {
        alert(`Invalid JSON: ${error.message}`);
        return;
      }
    }
    setIsEditing(!isEditing);
  };

  const handleJsonChange = (e) => {
    setEditableJson(e.target.value);
  };

  return (
    <div className="app-container">
      <div className="main-content">
        {/* Patient sidebar */}
        <div className="patient-sidebar">
          <center>
            <img src="/logo.png" alt="Atenta Logo" className="logo" />
            <button
              className="action-button"
              onClick={toggleRecording}
              disabled={processing}
            >
              <img
                src={micStatus === "on" ? "/micon.png" : "/micoff.png"}
                alt={micStatus === "on" ? "Microphone On" : "Microphone Off"}
                className="mic"
              />
              <span className="mic-status">
                {micStatus === "on"
                  ? "Recording... (Tap to Stop)"
                  : processing
                    ? "Processing..."
                    : "Tap to Record"}
              </span>
            </button>
          </center>
        </div>

        {/* Document viewer */}
        <div className="document-viewer">
          <div className="scrollable-content">
            <h2>Clinical Content: </h2>
            <div className={`status-badge ${status}`}>
              {status.toUpperCase()}
            </div>

            {error ? (
              <div className="error-message">
                {error}
                <button onClick={() => setError(null)}>Try Again</button>
              </div>
            ) : processing ? (
              <div className="processing-screen">
                <div className="loader"></div>
                <p>Processing recording...</p>
              </div>
            ) : jsonData ? (
              isEditing ? (
                <div className="json-editor-container">
                  <textarea
                    value={editableJson}
                    onChange={handleJsonChange}
                    className="json-editor"
                    spellCheck="false"
                  />
                  <div className="edit-controls">
                    <button onClick={handleEditToggle} className="save-button">
                      Save Changes
                    </button>
                    <button
                      onClick={() => {
                        setEditableJson(JSON.stringify(jsonData, null, 2));
                        setIsEditing(false);
                      }}
                      className="cancel-button"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="json-viewer">
                  <pre>{editableJson}</pre>

                  <button
                    onClick={handleEditToggle}
                    className={`edit-button ${status === "edited" ? "active" : ""}`}
                  >
                    {status === "edited" ? "✎ Edit Again" : "Edit"}
                  </button>
                </div>
              )
            ) : (
              <div className="empty-state">
                <p>No clinical data available</p>
                <p>Tap the microphone to record your notes</p>
              </div>
            )}
          </div>

          {jsonData && (
            <div className="button-container">
              <button
                className={`approve-button ${status === "approved" ? "active" : ""}`}
                onClick={handleApprove}
              >
                {status === "approved" ? "✓ Approved" : "Approve"}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
