import { useRef, useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import "../css/VideoStream.css" 

function VideoStream(props) {
    const [vidDeviceId, setVidDeviceId, setRecText, setCameraErr] = [
        props.vidDeviceId, 
        props.setVidDeviceId, 
        props.setRecText, 
        props.setCameraErr
    ];

    const [displaying, setDisplaying] = useState(false);
    const [detectionsCompleted, setDetectionsCompleted] = useState(false);

    const canvasRef = useRef(null);
    const videoRef = useRef(null);
    const virtualCanvasRef = useRef(null);
    const socketRef = useRef(null);
    
    useEffect(() => { // Connecting to websocket and starting clothing detection
        if (socketRef.current) {
            socketRef.current.close();
            console.log("Closing already open socket")
        } 

        const video = videoRef.current;
        const canvas = canvasRef.current;
        socketRef.current = new WebSocket("wss://outfit-detect-recs-production.up.railway.app/webcam/");
        
        startDetections(video, canvas);
    }, [])

    const displayDetections = async (video, canvas, imgBlob) => { // Displaying frames picked up from webcam except with detections/labels
        const ctx = canvas.getContext('2d');
        ctx.width = video.videoWidth;
        ctx.height = video.videoHeight;

        const frame = await createImageBitmap(imgBlob);
    
        ctx.drawImage(frame, 0, 0);
        if (!displaying) setDisplaying(true);
    }

    const startDetections = (video, canvas) => {
        let intervalId;
        let stream;

        socketRef.current.addEventListener('open', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: false, 
                    video: {
                        deviceId: vidDeviceId,
                        width: { max: 320 },
                        height: { max: 240 }
                    }
                });

                video.srcObject = stream;
                await video.play();

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                intervalId = setInterval(() => {
                    try {
                        const virtualCanvas = virtualCanvasRef.current;
                        virtualCanvas.width = video.videoWidth;
                        virtualCanvas.height = video.videoHeight;

                        const ctx = virtualCanvas.getContext('2d');
                        ctx.drawImage(video, 0, 0);
                    
                        if (socketRef.current.readyState === socketRef.current.OPEN) {
                            virtualCanvas.toBlob((blob) => socketRef.current.send(blob), 'image/jpeg'); // Sending frame bytes to back end                         
                        }

                    } catch (e) {
                        if (e instanceof TypeError) { // Occurs when there's an issue with camera
                            socketRef.current.close();
                            setCameraErr(true);
                            console.error(e)
                            clearInterval(intervalId);
                            
                        } else {
                            console.error(e)
                        }
                    }
                }, 70); 

            } catch (e) {
                if (e.message === "Could not start video source") {
                    console.log(e.message);
                    setCameraErr(true);
                    socketRef.current.close();

                } else {
                    console.log(e);
                }
            }
        });
        
        socketRef.current.addEventListener('message', (m) => {
            if (typeof m.data == "string") {
                if (m.data !== "Detections completed.") {
                    setDetectionsCompleted(true);
                    setRecText(m.data);
                    socketRef.current.close();
                }

            } else {
                displayDetections(video, canvas, m.data);                
            }
        });

        socketRef.current.addEventListener('close', async () => {
            clearInterval(intervalId);
            video.pause();

            // Stopping camera from recording
            if (stream) {
                stream.getTracks().forEach((track) => {
                    track.stop();
                });
            }

            setVidDeviceId(null);
        })
    };

    const endStream = () => {
        socketRef.current.close();
    }

    return(
        <div className="video-stream">    
            {displaying && !detectionsCompleted &&
                <p>Step back from camera to detect entire outfit. Video resolution reduced to decrease latency.</p>
            }

            <video ref={videoRef} id="video" style={{display: "none"}} />
            {!detectionsCompleted && <canvas ref={canvasRef} className="stream-canvas" />}
            <canvas ref={virtualCanvasRef} className='stream-virtual-canvas' style={{display: "none"}} />

            {displaying && !detectionsCompleted &&
                <div className='process-status'>
                    <div className="loading" />
                    <p>Detecting outfit</p>
                </div>    
            }                

            {displaying && !detectionsCompleted && <button className='btn' onClick={endStream}>Stop</button>}
        </div>
    );
}


VideoStream.propTypes = {
    vidDeviceId: PropTypes.string,
    setVidDeviceId: PropTypes.func,
    setRecText: PropTypes.func,
    setCameraErr: PropTypes.func
}


export default VideoStream;