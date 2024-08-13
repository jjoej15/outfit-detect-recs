import { useRef, useEffect, useState } from 'react';

function VideoStream() {
    const [isVirtualCanvas, setIsVirtualCanvas] = useState(true);
    const [videoDevices, setVideoDevices] = useState([]);
    const [vidDeviceId, setVidDeviceId] = useState(null);
     
    const canvasRef = useRef(null);
    const videoRef = useRef(null);
    const virtualCanvasRef = useRef(null);
    
    useEffect(() => { // Getting video devices
        const getDevices = async () => {
            const devices = await navigator.mediaDevices.enumerateDevices();
            let videoDevices = []
            for (const device of devices) {
                if (device.kind === 'videoinput' && device.deviceId) {
                    videoDevices.push({"deviceId": device.deviceId, "label": device.label});
                }
            }

            setVideoDevices(videoDevices);       
        }

        getDevices();
    }, []);

    useEffect(() => { // Connecting to websocket and starting clothin detection
        if (vidDeviceId) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const socket = new WebSocket("ws://localhost:8080/webcam/");
            
            startDetection(video, canvas, socket);
        }
    }, [vidDeviceId])

    const displayDetections = async (video, canvas, imgBlob) => {
        const ctx = canvas.getContext('2d');
        ctx.width = video.videoWidth;
        ctx.height = video.videoHeight;

        const frame = await createImageBitmap(imgBlob);
    
        ctx.drawImage(frame, 0, 0)
    }

    const startDetection = async (video, canvas, socket) => {
        let stream = await navigator.mediaDevices.getUserMedia({
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

        const intervalId = setInterval(() => {
            const virtualCanvas = virtualCanvasRef.current;
            virtualCanvas.width = video.videoWidth;
            virtualCanvas.height = video.videoHeight;

            const ctx = virtualCanvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            virtualCanvas.toBlob((blob) => socket.send(blob), 'image/jpeg');
        }, 42);

        socket.addEventListener('message', (m) => {
            displayDetections(video, canvas, m.data);
        })
    };

    return(
        <div className="video-stream">
            {videoDevices && videoDevices.map((d) => <button type='button' key={d.deviceId} onClick={() => setVidDeviceId(d.deviceId)}>{d.label}</button>)}
            {vidDeviceId && <video ref={videoRef} id="video" style={{display: "none"}} />}
            {vidDeviceId && <canvas ref={canvasRef} className="canvas" />}
            {isVirtualCanvas && <canvas ref={virtualCanvasRef} className='virtual-canvas' style={{display: "none"}} />}
        </div>
    );
}

export default VideoStream;