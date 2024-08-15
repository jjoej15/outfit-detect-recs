import { useRef, useEffect, useState } from 'react';
import PropTypes from 'prop-types';
 
function VideoStream(props) {
    const [vidDeviceId, setVidDeviceId] = [props.vidDeviceId, props.setVidDeviceId];
     
    const canvasRef = useRef(null);
    const videoRef = useRef(null);
    const virtualCanvasRef = useRef(null);
    const socket = new WebSocket("ws://localhost:8080/webcam/");
    
    useEffect(() => { // Connecting to websocket and starting clothing detection
        const video = videoRef.current;
        const canvas = canvasRef.current;
        // const socket = new WebSocket("ws://localhost:8080/webcam/");
        
        startDetections(video, canvas);
    }, [])

    const displayDetections = async (video, canvas, imgBlob) => {
        const ctx = canvas.getContext('2d');
        ctx.width = video.videoWidth;
        ctx.height = video.videoHeight;

        const frame = await createImageBitmap(imgBlob);
    
        ctx.drawImage(frame, 0, 0)
    }

    const startDetections = (video, canvas) => {
        let intervalId;
        let stream;

        socket.addEventListener('open', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: false, 
                    video: {
                        deviceId: vidDeviceId,
                        width: { max: 640 },
                        height: { max: 480 }
                    }
                });

                video.srcObject = stream;
                await video.play();

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                intervalId = setInterval(() => {
                    const virtualCanvas = virtualCanvasRef.current;
                    virtualCanvas.width = video.videoWidth;
                    virtualCanvas.height = video.videoHeight;

                    const ctx = virtualCanvas.getContext('2d');
                    ctx.drawImage(video, 0, 0);
                
                    if (socket.readyState === socket.OPEN) virtualCanvas.toBlob((blob) => socket.send(blob), 'image/jpeg');
                }, 70);                
            } catch (e) {
                if (e.message === "Could not start video source") {
                    console.log(e.message)
                    socket.close() 
                } else {
                    console.log(e)
                }
            }

        })
        
        socket.addEventListener('message', (m) => {
            displayDetections(video, canvas, m.data);
        })

        socket.addEventListener('close', async () => {
            clearInterval(intervalId);
            video.pause();
            if (stream) {
                stream.getTracks().forEach((track) => {
                    track.stop();
                });
            }
            setVidDeviceId(null);
        })
    };

    const endStream = () => {
        socket.close();
        // setVidDeviceId(null);
    }

    return(
        <div className="video-stream">
            <video ref={videoRef} id="video" style={{display: "none"}} />
            <canvas ref={canvasRef} className="canvas" />
            <canvas ref={virtualCanvasRef} className='virtual-canvas' style={{display: "none"}} />
            <button onClick={endStream}>Stop</button>
        </div>
    );
}


VideoStream.propTypes = {
    vidDeviceId: PropTypes.string,
    setVidDeviceId: PropTypes.func
}


export default VideoStream;