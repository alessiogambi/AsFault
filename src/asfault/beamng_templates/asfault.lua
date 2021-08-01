local M = {}

egoVehicleState = {}
egoVehicleState['steering'] = 0

egoGForce = {}
egoGForce['gx'] = 0
egoGForce['gy'] = 0
egoGForce['gz'] = 0

local helper = require('scenario/scenariohelper')
local socket = require('socket.socket')

local host = '{{host}}'
local port = {{port}}

local aiControlled = {{ai_controlled}}

local maxSpeed = {{max_speed}}
local speedFactor = 3
local naviGraph = {{navi_graph}}

local _log = log
local function log(level, message)
    _log(level, 'asfault', message)
end

log('I', 'Starting up AsFault Lua API')

local con = nil
local clients_read = {}
local clients_write = {}

local handlers = {}

local loaded = false
local hello = false
local started = false
local cameraSet = false
local goalSet = false
local startCooldown = 0

local frame = 0
local timeLeft = {{time_left}}
local raceStarted = false


local function receive(c)
    local line, err = c:receive()
    if err then
        log('E', 'Error whilst reading from socket: ' .. tostring(error))
    end

    log('D', 'Got line from socket: "' .. line .. '"')
    return line
end

local function readSocketMessage()
    local read, write, _ = socket.select(clients_read, clients_write, 0)

    local message = nil
    for _, c in ipairs(read) do
        if write[c] == nil then
            goto continue
        end

        c:settimeout(0.1, 't')

        message = receive(c)

        ::continue::
    end

    return message
end

local function sendSocketMessage(message)
    message = message.."\n"
    if con then
        con:send(message)
    end
end

local function evalTS(script)
    TorqueScript.eval(script)
end

local function startHandler(par)
    sendSocketMessage("START:true")
end

local function helloHandler(par)
    scenario_scenarios.changeState('running')
    guihooks.trigger('ChangeState', 'menu')

    if maxSpeed then
        -- be.physicsMaxSpeed = true
        be:setPhysicsDeterministic(true)
        be:setPhysicsSpeedFactor(speedFactor)
    else
        speedFactor = 1
    end

    if naviGraph then
        map.setDebugMode('graph')
    end
end

local function killHandler(par)
    log('I', 'Stop scenario')
    scenario_scenarios.stop()
    --    TorqueScript.eval('quit();')
end

-- called when countdown finished
local function onRaceStart()
    log('I', 'Race Start')
end

local function onClientPostStartMission()
    log('I', 'Connecting to AsFault socket on port ' .. port)
    con = socket.connect(host, port)
    table.insert(clients_read, con)
    table.insert(clients_write, con)
end

-- data { vehicleName = 'string', waypointName = 'string' }
local function onRaceWaypointReached(data)
end

local function isEgoVehiclePresent()
    local playerVec = be:getPlayerVehicleID(0)
    local carData = map.objects[playerVec]
    return carData ~= nil
end

local function checkEgoVehiclePresent()
    if isEgoVehiclePresent() then
        if started and not goalSet then
            if not cameraSet then
                core_camera.switchCamera(0, 1)
                -- core_camera.setByName(0, 'external')
                core_camera.setFOV(be:getPlayerVehicleID(0), 60)
                -- core_camera.setDistance(be:getPlayerVehicleID(0), 3)
                cameraSet = true
            end
            local rot = quatFromDir(vec3({{carDir.x}}, {{carDir.y}}))
            be:getPlayerVehicle(0):setPositionRotation({{pos.x}}, {{pos.y}}, {{pos.z}}, rot.x, rot.y, rot.z, rot.w)
            local arg = {
                vehicleName = 'egovehicle',
                waypoints = { {{waypoints}} },
                driveInLane = 'on',
                aggression = {{risk}},
                
                routeSpeed = {{speed_limit}},
                routeSpeedMode = '{{speed_limit_mode}}',

                resetLearning = true
            }
            if aiControlled then
                helper.setAiPath(arg)
            end
            goalSet = true
            log('I', 'Goal Set')
        end
    end
end

local function requestEgoVehicleData()
    -- Send the player vehicle a command to send back to this object a message with the state set to the value at the receiving end 
    local command = "obj:queueGameEngineLua('egoVehicleState[\"steering\"] = '..input.state.steering.val)"
    be:getPlayerVehicle(0):queueLuaCommand(command)
    
    -- Get the values of the GForce. There's should be a smarted way of doing so, but I cannot find it right now
    be:getPlayerVehicle(0):queueLuaCommand("obj:queueGameEngineLua('egoGForce[\"gx\"] = '..sensors.gx)")
    be:getPlayerVehicle(0):queueLuaCommand("obj:queueGameEngineLua('egoGForce[\"gy\"] = '..sensors.gy)")
    be:getPlayerVehicle(0):queueLuaCommand("obj:queueGameEngineLua('egoGForce[\"gz\"] = '..sensors.gz)")

end

local function handleSocketInput()
    local message = readSocketMessage()
    if message then
        local split = string.find(message, ':')
        if split ~= nil then
            local command = string.sub(message, 0, split - 1)
            local param = string.sub(message, split + 1)
            local handler = handlers[command]
            print('Got command: '..command)
            if handler then
                handler(param)
            end
        end
    end
end

local function produceSocketOutput()
    local playerVec = be:getPlayerVehicleID(0)
    if playerVec ~= -1 then
        local carData = map.objects[playerVec]
        if ( carData and goalSet ) then
            local message = "{"
            -- car state attributes
            message = message .. "\"pos_x\": " .. carData.pos.x .. ", "
            message = message .. "\"pos_y\": " .. carData.pos.y .. ", "
            message = message .. "\"pos_z\": " .. carData.pos.z .. ", "

            message = message .. "\"steering\": " .. egoVehicleState['steering'] .. ", "

            message = message .. "\"vel_x\": " .. carData.vel.x .. ", "
            message = message .. "\"vel_y\": " .. carData.vel.y .. ", "
            message = message .. "\"vel_z\": " .. carData.vel.z .. ", "

            message = message .. "\"g_x\": " .. egoGForce['gx'] .. ", "
            message = message .. "\"g_y\": " .. egoGForce['gy'] .. ", "
            message = message .. "\"g_z\": " .. egoGForce['gz'] .. ", "

            -- Additional data from the execution
            message = message .. "\"damage\": " .. carData.damage .. ", "
            -- Tag the message with the scenario/logical time if not nil
            if ( scenario_scenarios.getScenario() == nil or scenario_scenarios.getScenario().timer == nil ) then
                message = message .. "\"timestamp\": -1 "
            else
                message = message .. "\"timestamp\": " .. scenario_scenarios.getScenario().timer
            end

            -- Message termination
            message = message .. "}\n"

            --log('I', message)

            message = "STATE:" .. message

            sendSocketMessage(message)
        end
    end
end

-- called every 250ms. Use "status.setScenarioFailed()" for return failed test
-- data { dTime = number }
local function onRaceTick(raceTickTime)
    if con then
        handleSocketInput()
        if goalSet then
            produceSocketOutput()
        end
    end
    checkEgoVehiclePresent()
end

local function onUpdate(tickTime)
    timeLeft = timeLeft - tickTime
    if timeLeft <= 0 then
        -- killHandler()
    end

    if not started then
        handleSocketInput()
    end

    if startCooldown > 0 then
        startCooldown = startCooldown - 1
    end

    if loaded and not hello and startCooldown == 0 then
        sendSocketMessage("HELLO:true")
        hello = true
    end

    if started and isEgoVehiclePresent() then
        requestEgoVehicleData()
    end

    if raceStarted then
        frame = frame + 1
        if frame % math.floor(15 / speedFactor) == 0 then
            if con then
                handleSocketInput()
                if goalSet then
                    produceSocketOutput()
                end
            end
        end
    end
end

local function onScenarioUIReady()
    
end

local function onScenarioLoaded()
    loaded = true
    startCooldown = 120
end

local function onRaceInit()
    started = true
    log('I', 'Set started flag.')
    sendSocketMessage("START:true")
end

local function onCountdownEnded()
    raceStarted = true
    sendSocketMessage("RACESTART:true")
end

local function onInit()
    log('D', 'Setting up socket handlers.')
    handlers["START"] = startHandler
    handlers["EVALTS"] = evalTS
    handlers["HELLO"] = helloHandler
    handlers["KILL"] = killHandler
end

M.onInit = onInit
M.onUpdate = onUpdate
M.onClientPostStartMission = onClientPostStartMission
M.onRaceStart = onRaceStart
M.onRaceWaypointReached = onRaceWaypointReached
M.onRaceTick = onRaceTick
M.onScenarioLoaded = onScenarioLoaded
M.onRaceInit = onRaceInit
M.onCountdownEnded = onCountdownEnded
M.onScenarioUIReady = onScenarioUIReady

return M