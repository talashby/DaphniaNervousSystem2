
#include "NervousSystem.h"

#include "Neurons.h"
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <assert.h>
#include "ParallelPhysics/ServerProtocol.h"
#include "ParallelPhysics/ObserverClient.h"
#include <algorithm>
#include <utility>
#include <mutex>
#include <algorithm>
#include <xutility>

#define HIGH_PRECISION_STATS 1

// constants
constexpr uint32_t c_eyeGeneralizationNetworkOneSideSize = 5;
constexpr uint32_t c_reinforcementTransferNewnessPosZ = 10;

//
std::vector< std::vector< std::vector<ReinforcementTransferNeuron*> > > s_brain;

static NervousSystem* s_nervousSystem = nullptr;
static std::array<std::thread, 3> s_threads;
static std::array<std::pair<uint32_t, uint32_t>, s_threads.size()> s_threadNeurons; // [first neuron; last neuron)
static std::atomic<bool> s_isSimulationRunning = false;
static std::atomic<uint64_t> s_time = 0; // absolute universe time
static std::atomic<uint32_t> s_waitThreadsCount = 0; // thread synchronization variable

// stats
static std::mutex s_statisticsMutex;
static std::vector<uint32_t> s_timingsNervousSystemThreads;
static std::vector<uint32_t> s_tickTimeMusNervousSystemThreads;
std::atomic<uint32_t> s_status; // NervousSystemStatus



constexpr static int EYE_COLOR_NEURONS_NUM =  PPh::GetObserverEyeSize()*PPh::GetObserverEyeSize();
static std::array<std::array<SensoryNeuron, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetwork;
static std::array<SimpleAdderNeuron, 25> s_eyeGeneralizationNetwork;
static std::array<ReinforcementTransferNeuron, s_eyeGeneralizationNetwork.size() /* same size */> s_reinforcementTransferNewnessNetwork;
static std::array<ReinforcementSourceNeuron, s_reinforcementTransferNewnessNetwork.size() /* same size */> s_ReinforcementSourceNewnessNetwork;
static std::array<PremotorNeuron, s_eyeGeneralizationNetwork.size() /* same size */> s_premotorNewnessNetwork;
static std::array<ReinforcementTransferNeuron, s_eyeGeneralizationNetwork.size() /* same size */> s_reinforcementTransferHungerNetwork;
static EmptinessActivatorNeuron s_emptinessActivatorNeuron;
static PremotorNeuron s_emptinessPremotorNeuron;
static std::array<MotorNeuron, 3> s_motorNetwork; // 0 - forward, 1 - left, 2 - right


struct NetworksMetadata
{
	uint32_t m_beginNeuronNum;
	uint32_t m_endNeuronNum; // last+1
	uint64_t m_begin;
	uint64_t m_end;
	uint32_t m_size;
};
static std::array<NetworksMetadata, 8> s_networksMetadata{
	NetworksMetadata{0, 0, (uint64_t)(&s_eyeNetwork[0][0]), (uint64_t)(&s_eyeNetwork[0][0]+EYE_COLOR_NEURONS_NUM), sizeof(SensoryNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_eyeGeneralizationNetwork[0], (uint64_t)(&s_eyeGeneralizationNetwork[0] + s_eyeGeneralizationNetwork.size()), sizeof(SimpleAdderNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_reinforcementTransferNewnessNetwork[0], (uint64_t)(&s_reinforcementTransferNewnessNetwork[0] + s_eyeGeneralizationNetwork.size()), sizeof(ReinforcementTransferNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_premotorNewnessNetwork[0], (uint64_t)(&s_premotorNewnessNetwork[0] + s_eyeGeneralizationNetwork.size()), sizeof(PremotorNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_reinforcementTransferHungerNetwork[0], (uint64_t)(&s_reinforcementTransferHungerNetwork[0] + s_reinforcementTransferHungerNetwork.size()), sizeof(ReinforcementTransferNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_emptinessActivatorNeuron, (uint64_t)(&s_emptinessActivatorNeuron + 1), sizeof(EmptinessActivatorNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_emptinessPremotorNeuron, (uint64_t)(&s_emptinessPremotorNeuron + 1), sizeof(PremotorNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_motorNetwork[0], (uint64_t)(&s_motorNetwork[0]+s_motorNetwork.size()), sizeof(MotorNeuron)}
};

namespace NSNamespace
{
	uint64_t GetNSTime()
	{
		return s_time;
	}

	uint32_t GetNeuronIndex(Neuron *neuron)
	{
		uint64_t neuronAddr = reinterpret_cast<uint64_t>(neuron);
		for (auto &metadata : s_networksMetadata)
		{
			if (neuronAddr >= metadata.m_begin && neuronAddr < metadata.m_end)
			{
				return (uint32_t)((neuronAddr - metadata.m_begin) / metadata.m_size);
			}
		}
		assert(false);
		return 0;
	}

	Neuron* GetNeuronInterface(uint32_t neuronId)
	{
		Neuron* neuron = nullptr;
		uint8_t neuronType = (neuronId >> 24) & 0xff;
		uint16_t neuronIndex = neuronId & 0xffffff;
		switch (neuronType)
		{
		case SensoryNeuron::GetTypeStatic():
		{
			assert(s_eyeNetwork.size());
			assert(neuronIndex < s_eyeNetwork.size()*s_eyeNetwork[0].size());
			uint16_t xx = neuronIndex / PPh::GetObserverEyeSize();
			uint16_t yy = neuronIndex - xx * PPh::GetObserverEyeSize();
			neuron = &s_eyeNetwork[xx][yy];
		}
		break;
		case MotorNeuron::GetTypeStatic():
			assert(neuronIndex < s_motorNetwork.size());
			neuron = &s_motorNetwork[neuronIndex];
			break;
		}
		assert(((uint64_t*)neuron)[0] > 0); // virtual table should exist
		return neuron;
	}

	ReinforcementTransferNeuron* GetReinforcementTransferNeuron(const PPh::VectorInt32Math &pos)
	{
		assert(pos.m_posX >= 0 && pos.m_posX < s_brain.size());
		assert(pos.m_posY >= 0 && pos.m_posY < s_brain[0].size());
		assert(pos.m_posZ >= 0 && pos.m_posZ < s_brain[0][0].size());
		return s_brain[pos.m_posX][pos.m_posY][pos.m_posZ];
	}

}

void NervousSystemThread(uint32_t threadNum);

void NervousSystem::Init()
{
	if (s_nervousSystem)
	{
		delete s_nervousSystem;
	}

#ifdef HIGH_PRECISION_STATS
	s_timingsNervousSystemThreads.resize(s_threads.size());
	s_tickTimeMusNervousSystemThreads.resize(s_threads.size());
#endif

	{ // fill neuron number fields in metadata
		uint32_t firstNeuronNum = 0;
		for (auto &el : s_networksMetadata)
		{
			el.m_beginNeuronNum = firstNeuronNum;
			el.m_endNeuronNum = firstNeuronNum + (uint32_t)((el.m_end - el.m_begin) / el.m_size);
			firstNeuronNum = el.m_endNeuronNum;
		}
	}
	uint32_t neuronsNum = s_networksMetadata[s_networksMetadata.size() - 1].m_endNeuronNum;
	assert(neuronsNum > 0);
	{
		uint32_t net3dSize = s_reinforcementTransferNewnessNetwork.size() + 2;
		s_brain.resize(net3dSize);
		for (auto &yArray : s_brain)
		{
			yArray.resize(net3dSize);
			for (auto &zArray : yArray)
			{
				zArray.resize(net3dSize);
			}
		}
	}

	{ // fill s_threadNeurons
		assert(s_threads.size() > 1);
		uint32_t threadsNum = s_threads.size();
		uint32_t step = neuronsNum / threadsNum;
		uint32_t remain = neuronsNum - step * threadsNum;
		uint32_t posBegin = 0;
		for (uint32_t ii = 0; ii < s_threadNeurons.size(); ++ii)
		{
			std::pair<uint32_t, uint32_t> &pair = s_threadNeurons[ii];
			int32_t posEnd = posBegin + step;
			if (0 < remain)
			{
				++posEnd;
				--remain;
			}
			pair.first = posBegin;
			pair.second = posEnd;
			posBegin = posEnd;
		}
	}

	{ // init eyeGeneralizationNetwork
		uint32_t yPos = 0;
		uint32_t index = 0;
		for (int ii = 0; ii < c_eyeGeneralizationNetworkOneSideSize; ++ii)
		{
			uint32_t xPos = 0;
			uint32_t yLength = 3;
			if (ii == 2)
			{
				yLength = 4;
			}
			for (int jj = 0; jj < c_eyeGeneralizationNetworkOneSideSize; ++jj)
			{
				uint32_t xLength = 3;
				if (jj == 2)
				{
					xLength = 4;
				}
				SynapseVector synapses = CreateSynapses(xPos, yPos, xLength, yLength);
				assert(index < s_eyeGeneralizationNetwork.size());
				s_eyeGeneralizationNetwork[index].InitExplicit(synapses);
				xPos += xLength;
				++index;
			}
			yPos += yLength;
		}
	}

	{	// init emptinessActivatorNeuron
		SynapseVector synapses;
		for (Neuron &neuron : s_eyeGeneralizationNetwork)
		{
			synapses.push_back(Synapse(&neuron));
		}
		s_emptinessActivatorNeuron.InitExplicit(synapses);
	}

	auto CreateMotorSynapses = []()
	{
		MotorSynapseVector motorSynapses;
		for (MotorNeuron &neuron : s_motorNetwork)
		{
			motorSynapses.push_back(MotorSynapse(&neuron));
		}
		return motorSynapses;
	};

	{	// init emptiness premotor neuron
		SynapseVector synapses;
		synapses.push_back(Synapse(&s_emptinessActivatorNeuron));
		s_emptinessPremotorNeuron.InitExplicit(std::move(synapses), CreateMotorSynapses());
	}

	{	// init reinforcement transfer newness network
		for (uint32_t ii = 0; ii < s_reinforcementTransferNewnessNetwork.size(); ++ii)
		{
			SynapseVector synapses;
			synapses.push_back(Synapse(&s_reinforcementTransferNewnessNetwork[ii]));
			s_premotorNewnessNetwork[ii].InitExplicit(std::move(synapses), CreateMotorSynapses());
			uint32_t xx = 1 + ii / c_eyeGeneralizationNetworkOneSideSize;
			uint32_t yy = 1 + ii % c_eyeGeneralizationNetworkOneSideSize;
			s_reinforcementTransferNewnessNetwork[ii].InitExplicit(&s_eyeGeneralizationNetwork[ii], &s_premotorNewnessNetwork[ii], PPh::VectorInt32Math(xx, yy, c_reinforcementTransferNewnessPosZ));
			s_brain[xx][yy][c_reinforcementTransferNewnessPosZ] = &s_reinforcementTransferNewnessNetwork[ii];
		}
	}

	{	// init Reinforcement Source newness network
		for (uint32_t ii = 0; ii < s_ReinforcementSourceNewnessNetwork.size(); ++ii)
		{
			s_ReinforcementSourceNewnessNetwork[ii].InitExplicit(&s_reinforcementTransferNewnessNetwork[ii], 50'000);
		}
	}

	s_nervousSystem = new NervousSystem();
}

NervousSystem* NervousSystem::Instance()
{
	return s_nervousSystem;
}

void NervousSystem::StartSimulation(uint64_t timeOfTheUniverse)
{
	s_isSimulationRunning = true;
	s_time = timeOfTheUniverse;
	m_lastTime = PPh::GetTimeMs();
	m_lastTimeUniverse = timeOfTheUniverse;

	for (const auto &metadata : s_networksMetadata)
	{
		for (uint32_t jj = metadata.m_beginNeuronNum; jj < metadata.m_endNeuronNum; ++jj)
		{
			Neuron *neuron = reinterpret_cast<Neuron*>(metadata.m_begin + (jj - metadata.m_beginNeuronNum) * metadata.m_size);
			neuron->Init();
		}
	}
	s_waitThreadsCount = (uint32_t)s_threads.size();
	for (uint16_t ii = 0; ii < s_threads.size(); ++ii)
	{
		s_threads[ii] = std::thread(NervousSystemThread, ii);
	}
}

void NervousSystem::StopSimulation()
{
	if (!s_isSimulationRunning)
	{
		return;
	}
	s_isSimulationRunning = false;
	for (uint16_t ii = 0; ii < s_threads.size(); ++ii)
	{
		s_threads[ii].join();
	}
}

bool NervousSystem::IsSimulationRunning() const
{
	return s_isSimulationRunning;
}

void NervousSystem::GetStatisticsParams(uint32_t &reinforcementLevelStat, uint32_t &reinforcementsCountStat, uint32_t &condReflCountStat,
	uint32_t &movingSpontaneousCount, uint32_t &condReflLaunched, int32_t &minConditionedTmp,
	uint32_t &minNervousSystemTiming, uint32_t &maxNervousSystemTiming,	uint32_t &conditionedReflexCreatorTiming,
	uint32_t &condReflPredictionResult) const
{
	std::lock_guard<std::mutex> guard(s_statisticsMutex);
	minNervousSystemTiming = std::numeric_limits<uint32_t>::max();
	maxNervousSystemTiming = 0;
	for (uint32_t ii = 0; ii < s_tickTimeMusNervousSystemThreads.size()-1; ++ii)
	{
		minNervousSystemTiming = std::min(s_tickTimeMusNervousSystemThreads[ii], minNervousSystemTiming);
		maxNervousSystemTiming = std::max(s_tickTimeMusNervousSystemThreads[ii], maxNervousSystemTiming);
	}
	conditionedReflexCreatorTiming = s_tickTimeMusNervousSystemThreads[s_tickTimeMusNervousSystemThreads.size() - 1];
}

uint64_t NervousSystem::GetTime() const
{
	return s_time;
}

void NervousSystem::NextTick(uint64_t timeOfTheUniverse)
{
	while(s_waitThreadsCount) // Is Tick Finished
	{ }
	{
		if (timeOfTheUniverse > s_time)
		{
			s_waitThreadsCount = (uint32_t)s_threads.size();

			if (PPh::GetTimeMs() - m_lastTime >= 1000 && s_time != m_lastTimeUniverse)
			{
				m_quantumOfTimePerSecond = (uint32_t)(s_time - m_lastTimeUniverse);
#ifdef HIGH_PRECISION_STATS
				std::lock_guard<std::mutex> guard(s_statisticsMutex);
				for (uint32_t ii = 0; ii < s_timingsNervousSystemThreads.size(); ++ii)
				{
					if (s_timingsNervousSystemThreads[ii] > 0)
					{
						s_tickTimeMusNervousSystemThreads[ii] = s_timingsNervousSystemThreads[ii] / m_quantumOfTimePerSecond;
						s_timingsNervousSystemThreads[ii] = 0;
					}
				}
#endif
				m_lastTime = PPh::GetTimeMs();
				m_lastTimeUniverse = s_time;
			}

			++s_time;
		}
	}
}

void NervousSystem::PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color)
{
	if (m_color.m_colorR != m_color.m_colorG || m_color.m_colorR != m_color.m_colorG)
	{
		uint8_t color = std::max(m_color.m_colorR, m_color.m_colorG);
		color = std::max(color, m_color.m_colorB);

		s_eyeNetwork[m_posX][m_posY].WriteDendrite(color);
		/*if (m_color.m_colorR > 0)
		{
			s_eyeNetworkRed[m_posX][m_posY].ExcitatorySynapse();
		}
		else if (m_color.m_colorG > 0)
		{
			s_eyeNetworkGreen[m_posX][m_posY].ExcitatorySynapse();
		}
		else if (m_color.m_colorB > 0)
		{
			s_eyeNetworkBlue[m_posX][m_posY].ExcitatorySynapse();
		}*/
	}
}

SynapseVector NervousSystem::CreateSynapses(uint32_t xPos, uint32_t yPos, uint32_t xLength, uint32_t yLength)
{
	SynapseVector synapses;
	for (uint32_t ii = yPos; ii < yPos+yLength; ++ii)
	{
		for (uint32_t jj = xPos; jj < xPos+xLength; ++jj)
		{
			synapses.push_back(Synapse(&s_eyeNetwork[jj][ii]));
		}
	}
	return synapses;
}

void NervousSystemThread(uint32_t threadNum)
{
	assert(s_threadNeurons.size() > threadNum);
	uint32_t beginNetworkMetadata = 0;
	for (const auto &metadata : s_networksMetadata)
	{
		if (s_threadNeurons[threadNum].first >= metadata.m_beginNeuronNum && s_threadNeurons[threadNum].first < metadata.m_endNeuronNum)
		{
			break;
		}
		++beginNetworkMetadata;
	}
	uint32_t endNetworkMetadata = 0;
	for (const auto &metadata : s_networksMetadata)
	{
		if (s_threadNeurons[threadNum].second >= metadata.m_beginNeuronNum && s_threadNeurons[threadNum].second < metadata.m_endNeuronNum)
		{
			++endNetworkMetadata;
			break;
		}
		++endNetworkMetadata;
	}
	assert(beginNetworkMetadata <= endNetworkMetadata);
	uint32_t endLocalNeuronNum = s_threadNeurons[threadNum].second;
	while (s_isSimulationRunning)
	{
#ifdef HIGH_PRECISION_STATS
		auto beginTime = std::chrono::high_resolution_clock::now();
#endif
		int32_t isTimeOdd = s_time % 2;
		uint32_t beginLocalNeuronNum = s_threadNeurons[threadNum].first;
		for (uint32_t ii = beginNetworkMetadata; ii < endNetworkMetadata; ++ii)
		{
			if (ii != beginNetworkMetadata)
			{
				beginLocalNeuronNum = s_networksMetadata[ii].m_beginNeuronNum;
			}
			for (uint32_t jj = beginLocalNeuronNum; jj < endLocalNeuronNum && jj < s_networksMetadata[ii].m_endNeuronNum; ++jj)
			{
				Neuron *neuron = reinterpret_cast<Neuron*>(s_networksMetadata[ii].m_begin + (jj - s_networksMetadata[ii].m_beginNeuronNum) * s_networksMetadata[ii].m_size);
				neuron->Tick();
			}
		}
#ifdef HIGH_PRECISION_STATS
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dif = endTime - beginTime;
		s_timingsNervousSystemThreads[threadNum] += (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(dif).count();
#endif
		--s_waitThreadsCount;
		while (s_time % 2 == isTimeOdd && s_isSimulationRunning)
		{
		}
	}
}
