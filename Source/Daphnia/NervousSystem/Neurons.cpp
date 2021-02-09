
#include "Neurons.h"

#include "NervousSystem.h"
#include "ParallelPhysics/ObserverClient.h"
#include <assert.h>

// constants
constexpr int32_t EXCITATION_ACCUMULATION_TIME = 100; // ms
constexpr uint16_t EXCITATION_ACCUMULATION_LIMIT = EXCITATION_ACCUMULATION_TIME * MILLISECOND_IN_QUANTS; // units
constexpr uint16_t SENSORY_NEURON_REINFORCEMENT_LIMIT = 65535; // units
constexpr uint32_t SENSORY_NEURON_REINFORCEMENT_REFRESH_TIME = 5 * SECOND_IN_QUANTS;  // quantum of time

//

Synapse::Synapse(Neuron *from)
{
	assert(from);
	m_from = from;
}

bool Synapse::IsActive() const
{
	assert(m_from);
	bool result = m_from->IsActive();
	return result;
}

uint32_t Synapse::ReadAxon() const
{
	assert(from);
	uint32_t result = m_from->ReadAxon();
	return result;
}

MotorSynapse::MotorSynapse(MotorNeuron *to)
{
	assert(to);
	m_to = to;
}

uint32_t MotorSynapse::GetWeight() const
{
	return m_weight;
}

bool MotorSynapse::IsActive() const
{
	assert(m_to);
	bool result = m_to->IsActive();
	return result;
}

void MotorSynapse::TransferIrritation(uint32_t irritation) const
{
	assert(m_to);
	m_to->AddIrritationToDendrite(irritation);
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
}

bool MotorNeuron::IsActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isActive[isTimeEven];
}

void MotorNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	uint32_t irritation = m_dendrite[isTimeOdd];
	m_dendrite[isTimeOdd] = 0;
	bool isActive = false;
	if (irritation)
	{
		isActive = true;
		uint32_t index = NSNamespace::GetNeuronIndex(this);
		switch (index)
		{
		case 0:
			PPh::ObserverClient::Instance()->SetIsForward(isActive);
			break;
		case 1:
			PPh::ObserverClient::Instance()->SetIsLeft(isActive);
			break;
		case 2:
			PPh::ObserverClient::Instance()->SetIsRight(isActive);
			break;
		}
	}

	m_isActive[isTimeOdd] = isActive;
}

void MotorNeuron::AddIrritationToDendrite(uint16_t addedIrritation)
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	m_dendrite[isTimeEven] += addedIrritation;
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
		}
	}

	assert(m_synapseIndex < m_synapses.size());
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	if (m_synapseIndex == m_synapses.size() - 1)
	{
		if (m_activityTime < m_activityTimeMax)
		{
			++m_activityTime;
			m_isActive[isTimeOdd] = true;
		}
		else
		{
			m_isActive[isTimeOdd] = false;
			if (m_inhibitionTime < m_inhibitionTimeMax)
			{
				++m_inhibitionTime;
			}
			else
			{
				m_activityTime = 0;
				m_inhibitionTime = 0;
			}
		}
	}
	else
	{
		m_isActive[isTimeOdd] = false;
	}
}

bool EmptinessActivatorNeuron::IsActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isActive[isTimeEven];
}

uint32_t EmptinessActivatorNeuron::ReadAxon() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isActive[isTimeEven] * 1000 * IRRITATION_MULTIPLIER;
}

void PremotorNeuron::InitExplicit(SynapseVector &synapses, MotorSynapseVector &motorSynapses)
{
	m_synapses = std::move(synapses);
	m_motorSynapses = std::move(motorSynapses);
}

void PremotorNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;

	bool isAllActive = true;
	uint32_t irritation = 0;
	for (const Synapse& synapse : m_synapses)
	{
		if (!synapse.IsActive())
		{
			isAllActive = false;
			break;
		}
		irritation += synapse.ReadAxon();
	}

	if (isAllActive)
	{
		bool activated = 0 == m_axon[isTimeOdd];
		if (activated)
		{ // select random motor neuron
			int32_t totalWeight = 0;
			for (const MotorSynapse& synapse : m_motorSynapses)
			{
				totalWeight += synapse.GetWeight();
			}
			int32_t rnd = PPh::Rand32(totalWeight);

			int32_t currentWeight = 0;
			uint32_t index = 0;
			for (const MotorSynapse& synapse : m_motorSynapses)
			{
				currentWeight += synapse.GetWeight();
				if (currentWeight > rnd)
				{
					m_activatedSynapseIndex = index;
					break;
				}
				++index;
			}
		}
	}

	if (irritation < m_axon[isTimeOdd])
	{
		if (m_axon[isTimeOdd] > FADING_VAL)
		{
			m_axon[isTimeOdd] -= FADING_VAL;
		}
		else
		{
			m_axon[isTimeOdd] = 0;
		}
	}
	else
	{
		m_axon[isTimeOdd] = irritation;
	}

	m_motorSynapses[m_activatedSynapseIndex].TransferIrritation(m_axon[isTimeOdd]);

}
