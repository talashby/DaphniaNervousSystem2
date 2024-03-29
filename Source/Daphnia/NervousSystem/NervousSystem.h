
#pragma once

#include "ParallelPhysics/PPhHelpers.h"
#include <vector>
#include <memory>

class Neuron;
class Synapse;
class ConditionedReflexCreatorNeuron;
class MotivationTransferNeuron;

typedef std::vector<Synapse> SynapseVector;

namespace NSNamespace
{
	uint32_t GetNeuronIndex(Neuron *neuron);
	uint64_t GetNSTime();
	Neuron* GetNeuronInterface(uint32_t neuronId);
	MotivationTransferNeuron* GetReinforcementTransferNeuron(const PPh::VectorInt32Math &pos);
}

class NervousSystem
{
public:
	static void Init();
	static NervousSystem* Instance();
	NervousSystem() = default;
	virtual ~NervousSystem() = default;

	void StartSimulation(uint64_t timeOfTheUniverse);
	void StopSimulation();
	bool IsSimulationRunning() const;
	void GetStatisticsParams(uint32_t &reinforcementLevelStat, uint32_t &reinforcementsCountStat, uint32_t &condReflCountStat,
		uint32_t &movingSpontaneousCount, uint32_t &condReflLaunched, int32_t &minConditionedTmp,
		uint32_t &minNervousSystemTiming, uint32_t &maxNervousSystemTiming, uint32_t &conditionedReflexCreatorTiming,
		uint32_t &condReflPredictionResult) const;
	uint64_t GetTime() const;
	void NextTick(uint64_t timeOfTheUniverse);

	void PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color);

private:

	static SynapseVector CreateSynapses(uint32_t xPos, uint32_t yPos, uint32_t xLength, uint32_t yLength);

	// statistics
	uint64_t m_lastTime;
	uint64_t m_lastTimeUniverse;
	uint32_t m_quantumOfTimePerSecond;
};

