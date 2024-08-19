import { useState, useEffect } from "react";
import VideoStream from "./VideoStream";
import circleExclamationPng from '../assets/circle-exclamation.png';
import '../css/UseCamera.css'

function UseCamera() {
    const [videoDevices, setVideoDevices] = useState([]);
    const [vidDeviceId, setVidDeviceId] = useState(null);
    const [recText, setRecText] = useState();
    const [noCamera, setNoCamera] = useState(false);

    useEffect(() => { // Getting video devices
        const getDevices = async () => {
            const devices = await navigator.mediaDevices.enumerateDevices();
            let videoDevices = []
            for (const device of devices) {
                if (device.kind === 'videoinput' && device.deviceId) {
                    videoDevices.push({"deviceId": device.deviceId, "label": device.label});
                }
            }
            
            if (videoDevices.length == 0) setNoCamera(true);
            else setVideoDevices(videoDevices);     
        }
    
        getDevices();
      }, []);

    return(
        <div className="use-camera">
            {noCamera && 
            <div className="no-cam-container">
                <div>
                    <img className="circle-img" src={circleExclamationPng}/>
                    <h3>No cameras found.</h3>
                    <p>
                        Ensure that permission for website to access camera is granted then reload page.
                    </p>                    
                </div>
            </div>
            }

            {videoDevices && !vidDeviceId &&
                <div className="device-select">
                    <h3>Select a camera device.</h3>

                    {videoDevices.map(
                        (d) => <button className="device-btn" key={d.deviceId} onClick={() => setVidDeviceId(d.deviceId)}>{d.label}</button>
                    )}
                </div>
            }
            {vidDeviceId && 
                <VideoStream vidDeviceId={vidDeviceId} setVidDeviceId={setVidDeviceId} setRecText={setRecText}/>
            }  
            {recText && <p>{recText}</p>}
        </div>
    );
}

export default UseCamera;