
#include "Neurons.h"

#include "NervousSystem.h"
#include "ParallelPhysics/ObserverClient.h"
#include <assert.h>
#include <limits>


// constants
constexpr int32_t EXCITATION_ACCUMULATION_TIME = 100; // ms
constexpr uint16_t EXCITATION_ACCUMULATION_LIMIT = EXCITATION_ACCUMULATION_TIME * MILLISECOND_IN_QUANTS; // units
constexpr uint16_t SENSORY_NEURON_REINFORCEMENT_LIMIT = 65535; // units
constexpr uint32_t SENSORY_NEURON_REINFORCEMENT_REFRESH_TIME = 5 * SECOND_IN_QUANTS;  // quantum of time
constexpr uint32_t TRANSFER_MOTIVATION_REDUCING = 500; // milli units
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

InhibitorSynapse::InhibitorSynapse(Neuron *to)
{
	assert(to);
	m_to = to;
}

void InhibitorSynapse::Inhibit(uint32_t val)
{
	m_to->Inhibit(val);
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

void MotorSynapse::AddWeight(uint32_t weight)
{
	m_weight += weight;
}

bool MotorSynapse::IsActive() const
{
	assert(m_to);
	bool result = m_to->IsActive();
	return result;
}

void MotorSynapse::TransferExcitation(uint32_t excitation) const
{
	assert(m_to);
	m_to->AddExcitationToDendrite(excitation);
}


void MotorSynapse::HalfWeight()
{
	m_weight /= 2;
	if (m_weight < EXCITATION_MULTIPLIER)
	{
		m_weight = EXCITATION_MULTIPLIER;
	}
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
	return ReadAxon() > 0;
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
	return m_axon[isTimeEven] * EXCITATION_MULTIPLIER;
}

void SensoryNeuron::WriteDendrite(uint8_t excitation)
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	m_dendrite[isTimeEven] = excitation;
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

void MotorNeuron::InitExplicit(const std::shared_ptr<InhibitorSynapse>& inhibitorSynapses)
{
	m_inhibitorSynapse = inhibitorSynapses;
}

bool MotorNeuron::IsActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isActive[isTimeEven];
}

void MotorNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	uint32_t inhibition = m_inhibitor[isTimeOdd];
	uint32_t excitation = m_dendrite[isTimeOdd];
	m_inhibitor[isTimeOdd] = 0;
	m_dendrite[isTimeOdd] = 0;
	bool isActive = false;
	if (excitation > inhibition)
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

	if (m_inhibitorSynapse)
	{
		m_inhibitorSynapse->Inhibit(excitation);
	}
}

void MotorNeuron::Inhibit(uint32_t addedInhibition)
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	m_inhibitor[isTimeEven] += addedInhibition;
}

void MotorNeuron::AddExcitationToDendrite(uint16_t addedExcitation)
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	m_dendrite[isTimeEven] += addedExcitation;
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
	return m_isActive[isTimeEven] * 1000 * EXCITATION_MULTIPLIER;
}

void PremotorNeuron::InitExplicit(SynapseVector &&synapses, MotorSynapseVector &&motorSynapses)
{
	assert(synapses.size());
	m_synapses = std::move(synapses);
	assert(motorSynapses.size());
	m_motorSynapses = std::move(motorSynapses);
}

void PremotorNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;

	bool isAllActive = true;
	uint32_t excitation = 0;
	for (const Synapse& synapse : m_synapses)
	{
		if (!synapse.IsActive())
		{
			isAllActive = false;
			break;
		}
		excitation += synapse.ReadAxon();
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

	if (excitation < m_axon[isTimeOdd])
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
		m_axon[isTimeOdd] = excitation;
	}

	m_motorSynapses[m_activatedSynapseIndex].TransferExcitation(m_axon[isTimeOdd]);
}

void PremotorNeuron::AddReinforcement(uint32_t reinforcement)
{
	if (IsActive() && m_motorSynapses[m_activatedSynapseIndex].IsActive())
	{
		m_motorSynapses[m_activatedSynapseIndex].AddWeight(reinforcement);
		if (m_motorSynapses[m_activatedSynapseIndex].GetWeight() > std::numeric_limits<uint32_t>::max() / 2)
		{
			for (MotorSynapse &synapse : m_motorSynapses)
			{
				synapse.HalfWeight();
			}
		}
	}
}

void MotivationTransferNeuron::InitExplicit(const Neuron *reinforceActivator, PremotorNeuron *premotorNeuron, const PPh::VectorInt32Math &pos3D)
{
	m_reinforcementActivator = reinforceActivator;
	m_premotorNeuron = premotorNeuron;
	m_pos3D = pos3D;
	m_transferMotivation[0].fill({});
	m_transferMotivation[1].fill({});
}

uint32_t GetCellTransferIndex(const PPh::VectorInt32Math &unitVector)
{
	int32_t index = (unitVector.m_posX + 1) * 3 + (unitVector.m_posY + 1);
	if (index > 4)
	{
		--index;
	}
	return index;
}

PPh::VectorInt32Math GetUnitVectorFromCellTransferIndex(uint32_t index) // index [0;7]
{
	if (index > 3)
	{
		++index;
	}
	PPh::VectorInt32Math unitVector;
	unitVector.m_posX = index / 3 - 1;
	unitVector.m_posY = (index - (unitVector.m_posX + 1) * 3) - 1;
	unitVector.m_posZ = 0;
	return unitVector;
}

void MotivationTransferNeuron::Tick()
{
	assert(false);
	uint64_t time = NSNamespace::GetNSTime();
	int isTimeOdd = time % 2;

	// central motivation source
	if (m_CentralMotivationSource[isTimeOdd] > 0)
	{
		for (int ii = 0; ii < m_transferMotivation[0].size(); ++ii)
		{
			PPh::VectorInt32Math unitVectorToNeighbour = GetUnitVectorFromCellTransferIndex(ii);
			PPh::VectorInt32Math nbrPos = m_pos3D + unitVectorToNeighbour;
			MotivationTransferNeuron *neuron = NSNamespace::GetReinforcementTransferNeuron(nbrPos);
			if (neuron)
			{
				neuron->TransferMotivation(this, m_CentralMotivationSource[isTimeOdd]);
			}
		}
	}

	// transfer external motivation
	uint32_t wholeMotivation = 0;
	if (m_transferCentralMotivation[isTimeOdd] > 0)
	{
		wholeMotivation += m_transferCentralMotivation[isTimeOdd];
		for (int ii = 0; ii < m_transferMotivation[0].size(); ++ii)
		{
			PPh::VectorInt32Math unitVectorToNeighbour = GetUnitVectorFromCellTransferIndex(ii);
			PPh::VectorInt32Math nbrPos = m_pos3D + unitVectorToNeighbour;
			MotivationTransferNeuron *neuron = NSNamespace::GetReinforcementTransferNeuron(nbrPos);
			if (neuron)
			{
				neuron->TransferMotivation(this, m_transferCentralMotivation[isTimeOdd]);
			}
		}
	}
	for (int ii = 0; ii < m_transferMotivation[0].size(); ++ii)
	{
		uint32_t motivation = m_transferMotivation[isTimeOdd][ii];
		wholeMotivation += motivation;
		m_transferMotivation[isTimeOdd][ii] = 0;
		if (motivation > 0)
		{
			PPh::VectorInt32Math unitVectorToNeighbour = GetUnitVectorFromCellTransferIndex(ii);
			{
				{
					PPh::VectorInt32Math nbrPos = m_pos3D + unitVectorToNeighbour;
					MotivationTransferNeuron *neuron = NSNamespace::GetReinforcementTransferNeuron(nbrPos);
					if (neuron)
					{
						neuron->TransferMotivation(this, motivation);
					}
				}
				if (2 == abs(unitVectorToNeighbour.m_posX) + abs(unitVectorToNeighbour.m_posY))
				{ // diagonal
					{
						PPh::VectorInt32Math nbrPos = m_pos3D + PPh::VectorInt32Math(unitVectorToNeighbour.m_posX, 0, 0);
						MotivationTransferNeuron *neuron = NSNamespace::GetReinforcementTransferNeuron(nbrPos);
						if (neuron)
						{
							neuron->TransferMotivation(this, motivation);
						}
					}
					{
						PPh::VectorInt32Math nbrPos = m_pos3D + PPh::VectorInt32Math(0, unitVectorToNeighbour.m_posY, 0);
						MotivationTransferNeuron *neuron = NSNamespace::GetReinforcementTransferNeuron(nbrPos);
						if (neuron)
						{
							neuron->TransferMotivation(this, motivation);
						}
					}
				}
			}
		}
	}
	if (m_reinforcementActivator->IsActive())
	{ // send reinforcement
		std::vector<PremotorNeuron*> neighbourPremotorNeurons;
		for (int ii = 0; ii < m_transferMotivation[0].size(); ++ii)
		{
			PPh::VectorInt32Math unitVectorToNeighbour = GetUnitVectorFromCellTransferIndex(ii);
			PPh::VectorInt32Math nbrPos = m_pos3D + unitVectorToNeighbour;
			MotivationTransferNeuron *neuron = NSNamespace::GetReinforcementTransferNeuron(nbrPos);
			if (neuron && neuron->GetPremotorNeuron()->IsActive())
			{
				neighbourPremotorNeurons.push_back(neuron->GetPremotorNeuron());
			}
		}
		uint32_t devidendReinforcement = m_reinforcement;
		if (m_reinforcementActivator->ReadAxon() < m_reinforcement)
		{
			devidendReinforcement = m_reinforcementActivator->ReadAxon();
		}
		for (PremotorNeuron* premotorNeuron : neighbourPremotorNeurons)
		{
			premotorNeuron->AddReinforcement(devidendReinforcement / neighbourPremotorNeurons.size());
		}
		m_reinforcement -= devidendReinforcement;
	}
	else
	{ // increase reinforcement potential
		m_reinforcement += PPh::ProbabilisticDivision(wholeMotivation, SECOND_IN_QUANTS * 10);
		if (m_reinforcement > wholeMotivation)
		{
			m_reinforcement = wholeMotivation;
		}
	}
}

void MotivationTransferNeuron::SetCentralMotivationSource(uint32_t m_motivation)
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	m_CentralMotivationSource[isTimeEven] = m_motivation;
}

void MotivationTransferNeuron::TransferMotivation(MotivationTransferNeuron* neighbour, uint32_t motivation)
{
	assert((uint64_t)motivation * TRANSFER_MOTIVATION_REDUCING < std::numeric_limits<uint32_t>.max());
	PPh::VectorInt32Math unitVector = m_pos3D - neighbour->GetPosition();
	uint32_t transferIndex = GetCellTransferIndex(unitVector);

	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	m_transferMotivation[transferIndex][isTimeEven] += motivation * TRANSFER_MOTIVATION_REDUCING / 1000;
}

void MotivationTransferNeuron::TransferCentralMotivation(uint32_t m_motivation)
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	m_transferCentralMotivation[isTimeEven] = m_motivation;
}

void MotivationSourceNeuron::InitExplicit(MotivationTransferNeuron *reinforcementTransferNeuron, uint32_t motivation)
{
	m_reinforcementTransferNeuron = reinforcementTransferNeuron;
	m_motivation = motivation;
}

void MotivationSourceNeuron::Tick()
{
	m_reinforcementTransferNeuron->SetCentralMotivationSource(m_motivation);
}

