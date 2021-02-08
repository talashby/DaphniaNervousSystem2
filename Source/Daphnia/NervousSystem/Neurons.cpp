
#include "Neurons.h"

#include "NervousSystem.h"
#include "ParallelPhysics/ObserverClient.h"
#include <assert.h>

// constants
constexpr int32_t EXCITATION_ACCUMULATION_TIME = 100; // ms
constexpr uint16_t EXCITATION_ACCUMULATION_LIMIT = EXCITATION_ACCUMULATION_TIME * MILLISECOND_IN_QUANTS; // units
constexpr uint16_t SENSORY_NEURON_REINFORCEMENT_LIMIT = 65535; // units
constexpr uint32_t SENSORY_NEURON_REINFORCEMENT_REFRESH_TIME = 5 * SECOND_IN_QUANTS;  // quantum of time
constexpr uint32_t MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME = 15 * SECOND_IN_QUANTS; // quantum of time
constexpr uint32_t MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION = 500 * MILLISECOND_IN_QUANTS; // quantum of time
//

Synapse::Synapse(Neuron *from)
{
	assert(from);
	m_from = from;
}

uint32_t Synapse::IsActive() const
{
	assert(from);
	bool result = m_from->IsActive();
	return result;
}

uint32_t Synapse::ReadAxon() const
{
	assert(from);
	uint32_t result = m_from->ReadAxon();
	return result;
}

uint32_t Neuron::GetId()
{
	uint32_t id = NSNamespace::GetNeuronIndex(this);
	uint32_t idTypeMask = GetType() << 24;
	id |= idTypeMask;
	return id;
}

bool Neuron::IsActive() const
{
	assert(false);
	return false;
}

void SensoryNeuron::Init()
{
}

bool SensoryNeuron::IsActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_axon[isTimeEven] > 0;
}

uint32_t SensoryNeuron::ReadAxon() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_axon[isTimeEven];
}

void SensoryNeuron::WriteDendrite(uint8_t exitation)
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	m_dendrite[isTimeEven] = exitation;
}

void SensoryNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	m_axon[isTimeOdd] = m_dendrite[isTimeOdd];
	m_dendrite[isTimeOdd] = 0;
}

void SensoryNeuronRed::Tick()
{
	SensoryNeuron::Tick();
}

void SensoryNeuronGreen::Tick()
{
	SensoryNeuron::Tick();
}

void SensoryNeuronBlue::Tick()
{
	SensoryNeuron::Tick();
}

bool SensoryNeuronBlue::IsActive() const
{
	return SensoryNeuron::IsActive();
}

void MotorNeuron::Init()
{
	assert(NSNamespace::GetNSTime()); // server should send time already
	m_spontaneusActivityTimeStart = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME * (PPh::Rand32(100) + 50) / 100;
	m_lastExcitationTime = NSNamespace::GetNSTime();
}

bool MotorNeuron::IsActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isActive[isTimeEven];
}

void MotorNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	m_accumulatedExcitation = m_accumulatedExcitation + m_dendrite[isTimeOdd];
	m_dendrite[isTimeOdd] = 0;
	m_accumulatedExcitation = std::min(m_accumulatedExcitation, (uint16_t)254);
	m_axon[isTimeOdd] = static_cast<uint8_t>(m_accumulatedExcitation);
	bool isActive = false;
	if (m_accumulatedExcitation)
	{
		--m_accumulatedExcitation;
		isActive = true;
		m_lastExcitationTime = NSNamespace::GetNSTime();
	}

	else if (!m_accumulatedExcitation)
	{
		if (NSNamespace::GetNSTime() - m_lastExcitationTime > m_spontaneusActivityTimeStart)
		{
			++m_movingSpontaneousCount;
			++m_accumulatedExcitation;
			m_spontaneusActivityTimeFinishAbs = NSNamespace::GetNSTime() + MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION;
			m_spontaneusActivityTimeStart = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME * (PPh::Rand32(67) + 66) / 100;
			NervousSystem::Instance()->SetStatus(NervousSystem::NervousSystemStatus::SpontaneousActivity);
		}
		else if (NSNamespace::GetNSTime() < m_spontaneusActivityTimeFinishAbs)
		{
			++m_accumulatedExcitation;
			NervousSystem::Instance()->SetStatus(NervousSystem::NervousSystemStatus::SpontaneousActivity);
		}
		else if (NSNamespace::GetNSTime() == m_spontaneusActivityTimeFinishAbs)
		{
			NervousSystem::Instance()->SetStatus(NervousSystem::NervousSystemStatus::Relaxing);
		}
	}

	uint32_t index = NSNamespace::GetNeuronIndex(this);
	switch (index)
	{
	case 0:
		//PPh::ObserverClient::Instance()->SetIsForward(isActive);
		break;
	case 1:
		//PPh::ObserverClient::Instance()->SetIsLeft(isActive);
		break;
	case 2:
		//PPh::ObserverClient::Instance()->SetIsRight(isActive);
		break;
	}
	m_isActive[isTimeOdd] = isActive;
}

void MotorNeuron::ExcitatorySynapse()
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	++m_dendrite[isTimeEven];
}

std::atomic<uint32_t> MotorNeuron::m_movingSpontaneousCount;
uint32_t MotorNeuron::GetMovingSpontaneousCount()
{
	return m_movingSpontaneousCount.load();
}

void SimpleAdderNeuron::InitExplicit(SynapseVector &synapses)
{
	m_synapses = std::move(synapses);
}

uint32_t SimpleAdderNeuron::ReadAxon() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_axon[isTimeEven];
}

void SimpleAdderNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	uint16_t axon = 0;
	for (const Synapse& synapse : m_synapses)
	{
		axon += synapse.ReadAxon();
	}
	if (axon < m_axon[isTimeOdd])
	{
		if (m_axon[isTimeOdd] > 0)
		{
			--m_axon[isTimeOdd];
		}
	}
	else
	{
		m_axon[isTimeOdd] = axon;
	}
}

void EmptinessActivatorNeuron::InitExplicit(SynapseVector &synapses)
{
	assert(synapses->size());
	m_synapses = std::move(synapses);
}

void EmptinessActivatorNeuron::Tick()
{
	if (m_synapses[m_synapseIndex].ReadAxon())
	{
		m_synapseIndex = 0;
	}
	else
	{
		if (m_synapseIndex < m_synapses.size() - 1)
		{
			++m_synapseIndex;
		}
		else
		{
			for (const Synapse& synapse : m_synapses)
			{
				if (synapse.ReadAxon())
				{
					m_synapseIndex = 0;
					break;
				}
			}
			if (m_synapseIndex == m_synapses.size() - 1)
			{
				int eee = 0;
			}
		}
	}

	assert(m_synapseIndex < m_synapses.size());
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	m_isActive[isTimeOdd] = m_synapseIndex == m_synapses.size() - 1;
}

bool EmptinessActivatorNeuron::IsActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isActive[isTimeEven];
}

void PremotorNeuron::InitExplicit(SynapseVector & synapses)
{
}

void PremotorNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;

	bool isAllActive = true;
	for (const Synapse& synapse : m_synapses)
	{
		if (!synapse.IsActive())
		{
			isAllActive = false;
			break;
		}
	}

	if (isAllActive)
	{
		bool activated = 0 == m_axon[isTimeOdd];
		if (activated)
		{ // select random motor neuron
			int32_t totalWeight = 0;
			for (const Synapse& synapse : m_motorSynapses)
			{
				totalWeight += synapse.GetWeight();
			}
			int32_t rnd = PPh::Rand32(totalWeight);

			int32_t currentWeight = 0;
			for (const Synapse& synapse : m_motorSynapses)
			{
				currentWeight += synapse.GetWeight();
				if (currentWeight > rnd)
				{
					m_activatedSynapse = synapse;
					break;
				}
			}
		}

		uint16_t axon = 0;
		for (const Synapse& synapse : m_synapses)
		{
			axon += synapse.ReadAxon();
		}

		if (axon < m_axon[isTimeOdd])
		{
			if (m_axon[isTimeOdd] > 0)
			{
				--m_axon[isTimeOdd];
			}
		}
		else
		{
			m_axon[isTimeOdd] = axon;
		}

		if (activated)
		{ // try to activate random motor neuron

		}
	}
}
